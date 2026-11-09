"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

_DEADLINE_START = _time.time()
_MAX_WALL = 140.0  # Safety margin before the 150s kill


def _check_deadline(fallback):
    """If we're getting close to the wall-clock limit, return fallback immediately."""
    if _time.time() - _DEADLINE_START > _MAX_WALL:
        return fallback
    return None


def dispatch(ready_operations, time):
    # Check deadline first; if we're over, pick the first available (index 0)
    fallback = 0
    early = _check_deadline(fallback)
    if early is not None:
        return early

    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    # Extract features
    remaining_works = np.array([op["remaining_work"] for op in ready_operations], dtype=np.float64)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=np.float64)

    # Avoid division by zero
    remaining_ops_safe = np.where(remaining_ops > 0, remaining_ops, 1.0)
    avg_work = remaining_works / remaining_ops_safe

    min_durations = np.array([
        min(op["machine_options"].values()) if op["machine_options"] else 0.0
        for op in ready_operations
    ], dtype=np.float64)

    num_machines_available = np.array([
        len(op["machine_options"]) if op["machine_options"] else 0
        for op in ready_operations
    ], dtype=np.float64)

    # Compute slack-based urgency: operations with less slack relative to their work are more urgent
    # Slack proxy: (remaining_work - min_duration * remaining_ops) / remaining_ops
    # This captures how much "extra" work remains beyond the minimum possible duration
    slack = (remaining_works - min_durations * remaining_ops_safe) / remaining_ops_safe

    def _norm(x):
        mn = np.min(x)
        mx = np.max(x)
        if mx - mn < 1e-12:
            return np.zeros_like(x)
        return (x - mn) / (mx - mn)

    # Normalized features
    nw = _norm(remaining_works)
    no = _norm(remaining_ops)
    naw = _norm(avg_work)
    md_inv = 1.0 - _norm(min_durations)
    nm = _norm(num_machines_available)
    ns = _norm(slack)

    # Score: prioritize operations with high remaining work, high density,
    # high urgency (low slack), and short min duration
    # The slack term captures bottleneck risk: operations close to their minimum
    # duration have less scheduling flexibility
    score = 3.0 * nw + 1.5 * naw + 0.5 * no + 0.5 * md_inv + 0.1 * nm + 1.0 * (1.0 - ns)

    best_idx = int(np.argmax(score))

    return best_idx
# EVOLVE-BLOCK-END
