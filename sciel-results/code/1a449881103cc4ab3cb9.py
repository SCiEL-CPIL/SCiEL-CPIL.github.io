"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import random
import math
import heapq

# Deadline guard
_START_TIME = None
_DEADLINE = 145.0  # Safety margin

def dispatch(ready_operations, time):
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = _time.time()
    
    # Deadline check
    elapsed = _time.time() - _START_TIME
    if elapsed > _DEADLINE:
        # Fallback: MWR
        if not ready_operations:
            return 0
        return int(max(range(len(ready_operations)), key=lambda i: ready_operations[i].get("remaining_work", 0)))
    
    if not ready_operations:
        return 0
    
    # If only one op, return it
    if len(ready_operations) == 1:
        return 0
    
    # Calculate scores for all ops
    # Heuristic: Balance remaining work (urgency) with processing time efficiency.
    # For makespan, prioritizing jobs with more remaining work is crucial to avoid
    # having a single long job left at the end which extends the makespan significantly.
    # However, if that job's next operation is very long on all machines, it might be
    # better to clear a shorter operation from another job first if that other job
    # has significant remaining work, to keep the pipeline moving?
    # Actually, standard "Critical Ratio" or "Longest Remaining Processing Time" (LRPT)
    # usually works well.
    # Let's try a score that is primarily driven by remaining_work, but slightly
    # penalized by the duration of the current operation (min duration).
    # Score = remaining_work - alpha * min_duration
    # Or Score = remaining_work / (1 + min_duration)
    
    # Let's try: Score = remaining_work / (min_duration + 1)
    # This heavily favors high remaining work, but if two jobs have similar remaining work,
    # it favors the one with shorter current operation (faster completion of this step).
    
    best_score = -1
    best_idx = 0
    
    for i, op in enumerate(ready_operations):
        rw = op.get("remaining_work", 0)
        mo = op.get("machine_options", {})
        
        if mo:
            durations = list(mo.values())
            min_d = min(durations)
        else:
            min_d = 1.0
            
        if min_d < 0:
            min_d = 0
            
        # Score formulation
        # We want to maximize this.
        # High rw is good.
        # Low min_d is good (frees up machine sooner? or just finishes step sooner).
        
        # Let's try a simple ratio: rw / (min_d + 1)
        # This is similar to the previous attempt but with +1 to avoid extreme sensitivity to small durations.
        score = rw / (min_d + 1.0)
        
        # Tie-breaker: if scores are very close, prefer higher remaining_work
        # Or just stick to the score.
        
        if score > best_score:
            best_score = score
            best_idx = i
            
    return int(best_idx)

# EVOLVE-BLOCK-END
