"""Rewritten dispatch rule for job-shop scheduling with stronger heuristics."""
# EVOLVE-BLOCK-START
import time as _time
import random as _random

_START = _time.monotonic()
_DEADLINE = _START + 145.0  # 5s safety margin

def _deadline_exceeded():
    return _time.monotonic() > _DEADLINE

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
    
    n = len(ready_operations)
    if n == 1:
        return 0
    
    # Extract features for each ready operation
    remaining_works = []
    remaining_ops = []
    min_durs = []
    avg_durs = []
    
    for op in ready_operations:
        remaining_works.append(op["remaining_work"])
        remaining_ops.append(op["remaining_ops"])
        
        opts = op["machine_options"]
        if isinstance(opts, dict):
            vals = list(opts.values())
        elif isinstance(opts, (list, tuple)):
            vals = list(opts)
        else:
            vals = [0]
        
        if vals:
            min_durs.append(min(vals))
            avg_durs.append(sum(vals) / len(vals))
        else:
            min_durs.append(0)
            avg_durs.append(0)
    
    # Compute normalized features
    def normalize(arr):
        if not arr:
            return []
        mn = min(arr)
        mx = max(arr)
        if mx - mn < 1e-12:
            return [0.0] * len(arr)
        return [(x - mn) / (mx - mn) for x in arr]
    
    # Normalize features
    nw = normalize(remaining_works)
    no = normalize(remaining_ops)
    nd = normalize(min_durs)
    nad = normalize(avg_durs)
    
    # Multi-criteria scoring with tuned weights
    # Primary: prioritize jobs with more remaining work (longest job first)
    # Secondary: prioritize jobs with more remaining ops (urgency)
    # Tertiary: prefer shorter immediate processing time (SPT)
    # Quaternary: prefer operations with lower average duration
    
    # Weights tuned for job-shop scheduling
    w1 = 0.45  # remaining work
    w2 = 0.25  # remaining ops
    w3 = 0.20  # shortest processing time (inverted)
    w4 = 0.10  # average duration (inverted)
    
    scores = []
    for i in range(n):
        score = (
            w1 * nw[i] +
            w2 * no[i] +
            w3 * (1.0 - nd[i]) +
            w4 * (1.0 - nad[i])
        )
        scores.append(score)
    
    # Add tiny random tiebreaker
    best_score = -1
    best_idx = 0
    for i in range(n):
        s = scores[i] + _random.random() * 1e-8
        if s > best_score:
            best_score = s
            best_idx = i
    
    return best_idx
# EVOLVE-BLOCK-END
