"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

_DEADLINE_START = _time.time()
_MAX_WALL = 145.0  # Safety margin before the 150s kill


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

    # Strategy: Longest Remaining Processing Time (LRPT)
    # This heuristic prioritizes the job with the most total remaining work.
    # It is a standard and effective rule for minimizing makespan in open-shop scheduling.
    # Tie-breaker: Most remaining operations (to handle cases where work is equal).
    
    best_idx = 0
    best_work = ready_operations[0]["remaining_work"]
    best_ops = ready_operations[0]["remaining_ops"]

    # Iterate through all ready operations to find the one with the maximum remaining work
    for i in range(1, n):
        work = ready_operations[i]["remaining_work"]
        ops = ready_operations[i]["remaining_ops"]

        # Compare with small tolerance to avoid floating-point issues
        if work > best_work + 1e-9:
            best_idx = i
            best_work = work
            best_ops = ops
        elif abs(work - best_work) <= 1e-9:
            # Tie-break: prefer the job with more remaining operations
            if ops > best_ops:
                best_idx = i
                best_ops = ops

    return best_idx
# EVOLVE-BLOCK-END
