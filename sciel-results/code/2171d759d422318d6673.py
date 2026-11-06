"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time


# Deadline guard: keep total dispatch calls well within budget.
# We track wall-clock start of the first call and a per-call soft limit.
_start_time = None
_CALL_COUNT = 0
# Safety: if we've been running for > 100s, just return 0 quickly.
_DEADLINE = 100.0  # seconds


def _elapsed():
    global _start_time
    if _start_time is None:
        _start_time = _time.monotonic()
    return _time.monotonic() - _start_time


def _pick_op(ready_operations):
    """
    Dispatch rule: pick the ready operation that minimizes expected completion
    time considering machine load and job urgency.
    
    Strategy: For each ready operation, compute a score based on:
    1. The minimum duration across its machine options (shortest processing time)
    2. Remaining work of the job (urgency - more remaining work = higher priority)
    3. A tie-breaker using remaining ops count
    
    We use a weighted combination:
    - Primary: shortest processing time (SPT) to reduce machine idle time
    - Secondary: most remaining work (to avoid starving long jobs)
    - Tertiary: most remaining ops (to keep jobs progressing)
    """
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    scores = np.zeros(n, dtype=np.float64)
    for i, op in enumerate(ready_operations):
        # Minimum duration across machine options
        durations = list(op["machine_options"].values())
        min_dur = min(durations) if durations else 1.0
        mean_dur = sum(durations) / len(durations) if durations else 1.0
        
        rem_work = op["remaining_work"]
        rem_ops = op["remaining_ops"]
        
        # Score: we want to maximize this (higher = pick this op)
        # Weighted: prioritize shorter ops (SPT) but also consider remaining work
        # Use: -min_dur (prefer shorter) + alpha * rem_work (prefer more remaining work)
        # Normalize-ish: rem_work can be large, min_dur is smaller
        # Let's use a ratio-based approach:
        # score = rem_work / (min_dur + 1)  -> favors ops with more work relative to their duration
        # Actually for makespan, SPT is generally good. Let's try:
        # score = -min_dur + 0.1 * rem_work  (shorter ops first, slight urgency boost)
        
        # Better: use the concept of "critical ratio" or just SPT with urgency tiebreak
        # For job shop, a good heuristic is:
        # 1. Pick op with shortest processing time (on its best machine)
        # 2. Tie-break by most remaining work
        # 3. Tie-break by most remaining ops
        
        scores[i] = -min_dur * 100.0 + rem_work * 0.1 + rem_ops * 0.01
    
    return int(np.argmax(scores))


def dispatch(ready_operations, time):
    global _CALL_COUNT
    _CALL_COUNT += 1
    
    # Deadline guard
    if _elapsed() > _DEADLINE:
        return 0
    
    if not ready_operations:
        return 0
    
    return _pick_op(ready_operations)
# EVOLVE-BLOCK-END
