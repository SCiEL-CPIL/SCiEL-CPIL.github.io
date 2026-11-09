"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_START = _time.monotonic()
_DEADLINE = _START + 145.0  # 5s safety margin

def _deadline_exceeded():
    return _time.monotonic() > _DEADLINE

def _normalize(arr):
    """Normalize array to [0, 1] range."""
    arr = np.asarray(arr, dtype=np.float64)
    mn = arr.min()
    mx = arr.max()
    if mx == mn:
        return np.ones_like(arr) * 0.5
    return (arr - mn) / (mx - mn)

def dispatch(ready_operations, time):
    """
    Dispatch rule for job-shop scheduling.
    Returns index of the ready operation to start next.
    """
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    # Check deadline
    if _deadline_exceeded():
        return 0

    # Extract features
    remaining_work = np.array([op["remaining_work"] for op in ready_operations], dtype=np.float64)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=np.float64)
    
    # For each op, find the minimum duration among its machine options
    min_durations = np.array([
        min(op["machine_options"].values()) if op["machine_options"] else 1.0
        for op in ready_operations
    ], dtype=np.float64)

    # Also compute max duration to capture range
    max_durations = np.array([
        max(op["machine_options"].values()) if op["machine_options"] else 1.0
        for op in ready_operations
    ], dtype=np.float64)

    # Normalize features
    nw = _normalize(remaining_work)
    no = _normalize(remaining_ops)
    nd_min = _normalize(min_durations)
    nd_max = _normalize(max_durations)

    # Multi-criteria scoring:
    # - Higher remaining_work -> more urgent (urgency to complete)
    # - Higher remaining_ops -> keep in motion (flow time)
    # - Lower min_duration -> SPT (shortest processing time)
    # - Lower max_duration -> avoid long operations that block machines
    # - Also consider the ratio remaining_work / remaining_ops as average work per op
    
    # Compute average work per remaining op
    avg_work_per_op = remaining_work / np.maximum(remaining_ops, 1.0)
    naw = _normalize(avg_work_per_op)

    # Weights: balance urgency, flow, SPT, and work-per-op
    w_work = 0.25      # urgency
    w_ops = 0.30       # keep in motion
    w_spt = 0.25       # shortest processing time
    w_avg = 0.20       # average work per op (higher = more complex remaining)

    # Score: maximize
    # - remaining_work (urgency)
    # - remaining_ops (flow)
    # - (1 - nd_min) for SPT (lower min duration is better)
    # - naw (higher average work per op means more complex, prioritize?)
    #   Actually, for makespan minimization, we might want to prioritize
    #   operations that are on the critical path. Let's use a different approach.
    
    # Critical path estimate: remaining_work / max possible throughput
    # But we don't have machine availability. Let's use a simpler heuristic.
    
    # Combine: favor operations with high remaining work, high remaining ops,
    # and low processing times.
    score = (w_work * nw + 
             w_ops * no + 
             w_spt * (1.0 - nd_min) + 
             w_avg * (1.0 - naw))

    # Pick the operation with the highest score
    best_idx = int(np.argmax(score))
    return best_idx
# EVOLVE-BLOCK-END
