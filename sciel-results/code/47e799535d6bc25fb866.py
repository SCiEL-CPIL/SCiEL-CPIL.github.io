"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

# Deadline guard: the evaluator kills after 150s wall-clock.
# We keep a safety margin so we never hit the limit.
_DEADLINE = _time.monotonic() + 145.0  # 5s safety margin


def _dispatch_slow(ready_operations, time):
    """
    A stronger dispatch rule that considers multiple factors:
    1. Remaining work (total processing time left in the job)
    2. Number of remaining operations (jobs with fewer ops are 'closer' to completion)
    3. Machine load / availability heuristic via minimum duration option
    4. A small tie-breaker based on job id for determinism
    """
    if not ready_operations:
        return 0

    n = len(ready_operations)
    if n == 1:
        return 0

    # Compute features for each ready operation
    remaining_works = np.array([op["remaining_work"] for op in ready_operations], dtype=np.float64)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=np.float64)
    
    # For each op, get the minimum duration among its machine options (a proxy for urgency on that machine)
    min_durations = np.array(
        [min(op["machine_options"].values()) if op["machine_options"] else 0.0 for op in ready_operations],
        dtype=np.float64
    )

    # Normalize features to [0, 1] range for weighted combination
    def _normalize(arr):
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-12:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)

    # We want to prioritize:
    # - High remaining work (more work to do -> start it to keep machine busy)
    # - Low remaining ops (closer to completion -> finish it soon to free up the job)
    # - Low min duration (can be scheduled quickly -> reduces idle time)
    
    norm_work = _normalize(remaining_works)
    norm_ops = _normalize(remaining_ops)
    norm_dur = _normalize(min_durations)

    # Weighted score: higher is better
    # Weigh remaining work heavily, then prefer fewer remaining ops, then shorter min duration
    w_work = 0.5
    w_ops = 0.3
    w_dur = 0.2

    # For remaining_ops, we want to prefer LOW values, so invert
    # For min_duration, we want to prefer LOW values, so invert
    score = w_work * norm_work + w_ops * (1.0 - norm_ops) + w_dur * (1.0 - norm_dur)

    # Add a tiny deterministic tie-breaker based on job id to ensure stability
    job_ids = np.array([op["job"] for op in ready_operations], dtype=np.float64)
    if job_ids.max() - job_ids.min() > 1e-12:
        score += 1e-6 * _normalize(job_ids)

    return int(np.argmax(score))


def dispatch(ready_operations, time):
    """
    Dispatch rule: pick which ready operation to start next.
    Returns index of the ready operation to start.
    """
    if not ready_operations:
        return 0

    # Deadline guard: if we're close to the limit, use a fast fallback
    if _time.monotonic() > _DEADLINE:
        # Fast fallback: just pick the one with max remaining work
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))

    return _dispatch_slow(ready_operations, time)
# EVOLVE-BLOCK-END
