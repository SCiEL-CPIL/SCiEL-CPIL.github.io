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

    # For each ready operation, compute the minimum and maximum duration across machine options
    min_durations = np.array([
        min(op["machine_options"].values()) if op["machine_options"] else 0.0
        for op in ready_operations
    ], dtype=np.float64)
    
    max_durations = np.array([
        max(op["machine_options"].values()) if op["machine_options"] else 0.0
        for op in ready_operations
    ], dtype=np.float64)

    # Flexibility: how much choice exists (max - min). High flexibility = more options to avoid congestion.
    flexibility = max_durations - min_durations

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
    # Inverse min duration: prefer short minimum processing time (LRPT-like)
    md_inv = 1.0 - _norm(min_durations)
    # Flexibility: prefer operations with more machine choice (can be scheduled on less busy machines)
    flex = _norm(flexibility)

    # Score:
    # - High remaining work (LRPT core)
    # - High average work (density)
    # - Short min duration (process fast on best machine)
    # - High flexibility (more options to avoid bottlenecks)
    # Weighting: LRPT is dominant, but flexibility helps in open-shop where machine choice matters
    score = 4.0 * nw + 1.5 * naw + 0.8 * md_inv + 0.7 * flex

    best_idx = int(np.argmax(score))

    return best_idx
# EVOLVE-BLOCK-END
