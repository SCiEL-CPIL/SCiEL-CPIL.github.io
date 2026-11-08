"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import math

# Deadline guard: evaluator kills after 150s wall-clock; keep a safety margin.
_START = _time.monotonic()
_DEADLINE = _START + 145.0  # 5s safety margin

def _deadline_exceeded():
    return _time.monotonic() > _DEADLINE

def _fast_dispatch(ready_operations, time):
    """
    Fast heuristic dispatch rule for job-shop scheduling.
    
    Strategy: Multi-criteria scoring combining:
    1. Remaining work (longest job remaining) - primary
    2. Number of remaining operations (urgency)
    3. Strong reciprocal preference for short processing time (reduces idle time)
    
    The reciprocal form 1/min_dur gives much stronger weight to short operations
    than linear inversion, which is critical for makespan because finishing a
    short op quickly frees a machine and unblocks downstream operations.
    """
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0
    
    remaining_works = []
    remaining_ops_counts = []
    min_durs = []
    
    for op in ready_operations:
        remaining_works.append(op.get("remaining_work", 0))
        remaining_ops_counts.append(op.get("remaining_ops", 0))
        opts = op.get("machine_options", {})
        if isinstance(opts, dict) and opts:
            min_durs.append(min(opts.values()))
        elif isinstance(opts, (list, tuple)) and len(opts) > 0:
            min_durs.append(min(opts))
        else:
            min_durs.append(0)
    
    # Normalize each criterion to [0, 1]
    def normalize(arr):
        mn = min(arr)
        mx = max(arr)
        if mx - mn < 1e-12:
            return [0.5] * len(arr)
        return [(x - mn) / (mx - mn) for x in arr]
    
    # Criterion 1: More remaining work is better (priority to long jobs)
    c1 = normalize(remaining_works)
    
    # Criterion 2: More remaining operations = longer job, higher priority
    c2 = normalize(remaining_ops_counts)
    
    # Criterion 3: Reciprocal of min duration - stronger preference for short ops
    # Using 1/(min_dur + eps) gives much more weight to short operations
    # than linear inversion would
    eps = 1e-6
    recip_vals = [1.0 / (d + eps) if d > 0 else 1.0 / eps for d in min_durs]
    c3 = normalize(recip_vals)
    
    # Weighted combination
    # Emphasize short processing time (c3) heavily because it directly reduces
    # machine idle time and unblocks downstream operations
    w1, w2, w3 = 0.35, 0.15, 0.50
    scores = [w1 * c1[i] + w2 * c2[i] + w3 * c3[i] for i in range(n)]
    
    # Pick the best
    best_idx = 0
    best_score = scores[0]
    for i in range(1, n):
        if scores[i] > best_score:
            best_score = scores[i]
            best_idx = i
    
    return int(best_idx)

def dispatch(ready_operations, time):
    """
    Dispatch rule for job-shop scheduling.
    
    Args:
        ready_operations: list of dicts, each with keys:
            - job: job identifier
            - op: operation identifier  
            - machine_options: dict {machine: duration} or similar
            - remaining_ops: number of remaining operations in the job
            - remaining_work: total remaining processing time for the job
        time: current simulation time
    
    Returns:
        int: index of the ready operation to start next
    """
    if _deadline_exceeded():
        return 0
    
    if not ready_operations:
        return 0
    
    return _fast_dispatch(ready_operations, time)
# EVOLVE-BLOCK-END
