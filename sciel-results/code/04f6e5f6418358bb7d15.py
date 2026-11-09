"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import random
import math
import heapq

# Deadline guard
_START_TIME = None
_DEADLINE = 150.0

def _fast_dispatch(ready_operations, time):
    """
    Fast heuristic dispatch: Shortest Processing Time (SPT) with tie-breaker by Remaining Work (MWR).
    This is generally effective for minimizing makespan in job-shop problems by keeping machines busy with short tasks first.
    """
    if not ready_operations:
        return 0
    
    best_idx = 0
    best_min_d = float('inf')
    best_rw = -1.0
    
    for i, op in enumerate(ready_operations):
        mo = op.get("machine_options", {})
        if not mo:
            min_d = float('inf')
        else:
            # Find min duration
            min_d = min(mo.values())
        
        rw = op.get("remaining_work", 0)
        
        # Primary: Min Duration (SPT)
        # Secondary: Max Remaining Work (MWR)
        if min_d < best_min_d - 1e-9:
            best_min_d = min_d
            best_rw = rw
            best_idx = i
        elif abs(min_d - best_min_d) < 1e-9:
            if rw > best_rw:
                best_rw = rw
                best_idx = i
                
    return int(best_idx)

def _heuristic_score(ready_operations, time, idx):
    """
    More complex heuristic.
    Uses a weighted combination of Remaining Work (MWR) and Shortest Processing Time (SPT).
    Weights are tuned to balance urgency and efficiency.
    """
    if not ready_operations:
        return 0

    ops = ready_operations
    n = len(ops)
    
    # Pre-calculate features
    rws = [0.0] * n
    min_ds = [0.0] * n
    max_rw = 0.0
    min_min_d = float('inf')
    
    for i in range(n):
        op = ops[i]
        rw = op.get("remaining_work", 0)
        mo = op.get("machine_options", {})
        
        if mo:
            min_d = min(mo.values())
        else:
            min_d = 1.0
            
        rws[i] = rw
        min_ds[i] = min_d
        
        if rw > max_rw:
            max_rw = rw
        if min_d < min_min_d:
            min_min_d = min_d
            
    if max_rw == 0:
        max_rw = 1.0
    if min_min_d == 0 or min_min_d == float('inf'):
        min_min_d = 1.0
        
    best_idx = 0
    best_score = -1.0
    
    # Weights: 
    # w1 for MWR (urgency), w2 for SPT (efficiency)
    # SPT is often stronger for makespan in flexible job shop.
    # Let's try w1=0.4, w2=0.6
    w1 = 0.4
    w2 = 0.6
    
    for i in range(n):
        # Normalize MWR: higher is better
        # Normalize SPT: lower is better -> use 1/min_d or (min_min_d / min_d)
        # Using ratio to keep scale consistent
        mwr_score = rws[i] / max_rw
        spt_score = min_min_d / min_ds[i] if min_ds[i] > 0 else 0.0
        
        score = w1 * mwr_score + w2 * spt_score
        
        if score > best_score:
            best_score = score
            best_idx = i
        elif abs(score - best_score) < 1e-9:
            # Tie breaker: prefer higher MWR
            if rws[i] > rws[best_idx]:
                best_idx = i
                
    return int(best_idx)

def dispatch(ready_operations, time):
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = _time.time()
    
    # Deadline check
    elapsed = _time.time() - _START_TIME
    # If we are close to the limit, return a very fast default
    if elapsed > _DEADLINE - 5.0:
        return _fast_dispatch(ready_operations, time)
    
    try:
        # Use the more complex heuristic if we have time, but _fast_dispatch is also good.
        # Let's use _heuristic_score which is O(N) and robust.
        return _heuristic_score(ready_operations, time, 0)
    except Exception:
        return _fast_dispatch(ready_operations, time)

# EVOLVE-BLOCK-END
