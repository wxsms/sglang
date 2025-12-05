from dataclasses import asdict
from typing import Optional
import threading
import zmq
import logging

from sglang.srt.metrics.collector import SchedulerStats

logger = logging.getLogger(__name__)


class MetricsEventPublisher:
    def __init__(self, port: int, dp_rank: int, dp_router_port: Optional[int], leader_host: Optional[str]) -> None:
        self.ctx = zmq.Context()
        self.dp_rank = dp_rank
        self._router_thread = None
        self._router_stop = threading.Event()

        if dp_rank == 0:
            self.metrics_pub = self.ctx.socket(zmq.PUB)
            self.metrics_pub.bind(f"tcp://*:{port}")
            logger.info(f"Metrics ZMQ publisher initialized on port {port}")
            if dp_router_port is not None:
                self.metrics_dp_router = self.ctx.socket(zmq.ROUTER)
                self.metrics_dp_router.bind(f"tcp://*:{dp_router_port}")
                logger.debug(f"Metrics ZMQ DP router initialized on port {dp_router_port}")
                self._router_thread = threading.Thread(target=self._router_proxy_thread, daemon=True)
                self._router_thread.start()
        else:
            assert leader_host is not None and dp_router_port is not None, \
                "leader_host and dp_router_port must be provided for non-zero dp_rank"
            self.metrics_dp_router = self.ctx.socket(zmq.DEALER)
            self.metrics_dp_router.setsockopt(zmq.IDENTITY, f"dp_{dp_rank}".encode())
            self.metrics_dp_router.connect(f"tcp://{leader_host}:{dp_router_port}")

    def destroy(self) -> None:
        self._router_stop.set()
        self._router_thread.join()

    def send(self, stats: SchedulerStats):
        stats.dp_rank = self.dp_rank
        msg = asdict(stats)
        msg["dp_rank"] = self.dp_rank
        if self.dp_rank == 0:
            self.metrics_pub.send_json(msg)
        else:
            self.metrics_dp_router.send_json(msg)

    def _router_proxy_thread(self):
        while not self._router_stop.is_set():
            try:
                if self.metrics_dp_router.poll(1000):
                    _, message = self.metrics_dp_router.recv_multipart()
                    self.metrics_pub.send(message)
            except Exception as e:
                logger.error(f"Exception in metrics router proxy thread: {e}")
