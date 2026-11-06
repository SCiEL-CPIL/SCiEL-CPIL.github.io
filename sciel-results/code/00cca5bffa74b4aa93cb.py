"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np


# Global state to cache parameters and track time
_start_time = _time.time()
_DEADLINE = 150.0  # Safety margin before 150s kill

def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    # Deadline guard: if we're close to the limit, use a very fast fallback
    elapsed = _time.time() - _start_time
    if elapsed > _DEADLINE - 5.0:
        # Fallback: pick the operation with minimum duration
        best_index = 0
        min_dur = float('inf')
        for i, op in enumerate(ready_operations):
            machine_options = op.get("machine_options", {})
            if not machine_options:
                continue
            md = min(machine_options.values())
            if md < min_dur:
                min_dur = md
                best_index = i
        return int(best_index)

    # Structural change: Bottleneck mitigation via Workload and Constraint Tightness
    # Prioritize operations belonging to jobs with high remaining work (Longest Job First)
    # and high constraint tightness (few machine options).
    # This aims to clear the most critical, constrained jobs early to reduce
    # the risk of them becoming the bottleneck later.
    
    best_score = -float('inf')
    best_index = 0
    
    for i, op in enumerate(ready_operations):
        machine_options = op.get("machine_options", {})
        if not machine_options:
            continue
        
        # Constraint tightness: fewer machine options = more constrained
        num_options = len(machine_options)
        min_duration = min(machine_options.values())
        remaining_work = op.get("remaining_work", 0)
        
        # Score: 
        # 1. Prioritize high remaining work (clear big jobs)
        # 2. Weight by constraint tightness (constrained jobs are more critical)
        # 3. Small bonus for shorter duration to keep flow moving
        # We want to maximize this score.
        
        # If remaining_work is 0, fall back to tightness/speed
        if remaining_work == 0:
            score = (1.0 / num_options) + (1.0 / max(min_duration, 1e-9))
        else:
            # Weighted Longest Job First for constrained ops
            score = (remaining_work / num_options) + (1.0 / max(min_duration, 1e-9))
            
        if score > best_score:
            best_score = score
            best_index = i
            
    return int(best_index)
# EVOLVE-BLOCK-END
