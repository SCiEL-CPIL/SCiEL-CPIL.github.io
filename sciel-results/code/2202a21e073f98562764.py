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

    # Structural change: Bottleneck mitigation via constraint tightness
    # Prioritize operations that have fewer machine options (higher constraint tightness)
    # to prevent future starvation. Break ties by shortest processing time.
    
    best_score = -float('inf')
    best_index = 0
    
    for i, op in enumerate(ready_operations):
        machine_options = op.get("machine_options", {})
        if not machine_options:
            continue
        
        # Constraint tightness: fewer machine options = more constrained
        num_options = len(machine_options)
        min_duration = min(machine_options.values())
        
        # Score: prioritize low constraint (few options) and short duration
        # Use a composite score where lower is better for both
        # We want to maximize the score, so invert the terms
        # tightness_score = 1 / num_options (higher is more constrained, we want to address these first)
        # speed_score = 1 / min_duration (higher is faster)
        
        # Combined: prioritize constrained ops, then fast ops
        score = (1.0 / num_options) + (1.0 / max(min_duration, 1e-9))
        
        if score > best_score:
            best_score = score
            best_index = i
            
    return int(best_index)
# EVOLVE-BLOCK-END
