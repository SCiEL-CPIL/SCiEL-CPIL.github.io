"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

# Deadline guard: keep a safety margin before the 150s wall-clock limit.
_DEADLINE = _time.time() + 145.0  # 5s safety margin
_DEADLINE_CHECK_INTERVAL = 0.05   # seconds between checks


def _deadline_exceeded():
    return _time.time() >= _DEADLINE


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next in an event-driven open-shop simulation.
    
    Strategy: Composite scoring that blends:
    1. Remaining work (job completion priority)
    2. Minimum duration option (machine utilization / LPT-like)
    3. Number of remaining ops (job urgency)
    
    The score is a weighted composite:
      score = w1 * normalized_remaining_work + w2 * normalized_min_duration + w3 * normalized_remaining_ops
    
    We pick the operation with the highest composite score.
    """
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    # Deadline guard
    if _deadline_exceeded():
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))

    # Extract features
    remaining_works = np.array([op["remaining_work"] for op in ready_operations], dtype=np.float64)
    min_durations = np.array([min(op["machine_options"].values()) for op in ready_operations], dtype=np.float64)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=np.float64)

    # Normalize each feature to [0, 1] range
    def _normalize(arr):
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-12:
            return np.ones_like(arr)
        return (arr - mn) / (mx - mn)

    norm_work = _normalize(remaining_works)
    norm_dur = _normalize(min_durations)
    norm_ops = _normalize(remaining_ops)

    # Weights: prioritize remaining work, then shorter min duration (LPT-like), then fewer remaining ops
    w1 = 0.55  # remaining work
    w2 = 0.30  # min duration (lower is better, so we invert)
    w3 = 0.15  # remaining ops (lower is better, so we invert)

    # For duration and remaining ops, lower is better, so we invert
    score = w1 * norm_work + w2 * (1.0 - norm_dur) + w3 * (1.0 - norm_ops)

    # Deadline guard after computation
    if _deadline_exceeded():
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))

    return int(np.argmax(score))
# EVOLVE-BLOCK-END
