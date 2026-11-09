"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import random as _random
import numpy as np

# Deadline guard
_START = _time.monotonic()
_DEADLINE = _START + 145.0  # 5s safety margin before 150s kill

def _deadline_exceeded():
    return _time.monotonic() > _DEADLINE

def _fast_dispatch(ready_operations, time):
    """
    Fast heuristic dispatch rule for job-shop scheduling.
    
    Strategy: Multi-criteria scoring combining:
    1. Remaining work (longer jobs first) - primary
    2. Remaining operations (more ops = longer job, prioritize) 
    3. Minimum processing time (shorter is better to keep machines busy)
    
    Returns index of best ready operation to start.
    """
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0
    
    remaining_works = np.array([op["remaining_work"] for op in ready_operations], dtype=np.float64)
    remaining_ops_counts = np.array([op["remaining_ops"] for op in ready_operations], dtype=np.float64)
    
    # Compute minimum duration for each operation
    min_durs = np.array([
        min(op["machine_options"].values()) if op["machine_options"] else 0.0
        for op in ready_operations
    ], dtype=np.float64)
    
    # Normalize each criterion to [0, 1]
    def normalize(arr):
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-12:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)
    
    # Criterion 1: More remaining work is better (prioritize long jobs)
    c1 = normalize(remaining_works)
    
    # Criterion 2: More remaining operations = longer job, prioritize
    # (This is the key correction from inspiration 1)
    c2 = normalize(remaining_ops_counts)
    
    # Criterion 3: Shorter immediate processing time is better
    c3 = 1.0 - normalize(min_durs)
    
    # Weighted combination: emphasize remaining work and ops
    weights = np.array([0.45, 0.35, 0.20], dtype=np.float64)
    scores = weights[0] * c1 + weights[1] * c2 + weights[2] * c3
    
    # Add small random tiebreaker to avoid deterministic pitfalls
    scores += _random.random() * 1e-6
    
    return int(np.argmax(scores))

def dispatch(ready_operations, time):
    """
    Dispatch rule for job-shop scheduling.
    
    Args:
        ready_operations: list of dicts, each with keys:
            - job: job identifier
            - op: operation identifier  
            - machine_options: dict {machine: duration}
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
