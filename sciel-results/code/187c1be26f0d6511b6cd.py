import numpy as np

def dispatch(ready_operations, time):
    if not ready_operations:
        return 0 # Or raise, but 0 is safe fallback? No, empty list shouldn't happen.
        
    # 1. Select Operation
    # We want to maximize remaining_work.
    # Tie-breaker: minimize the minimum processing time available for that op.
    
    max_work = -1
    best_idx = 0
    best_min_dur = float('inf')
    
    for i, op in enumerate(ready_operations):
        work = op['remaining_work']
        # Calculate min duration for this op
        if op['machine_options']:
            min_dur = min(op['machine_options'].values())
        else:
            min_dur = float('inf')
            
        # Compare
        # If work is significantly larger, it wins.
        # If work is same (or close), smaller min_dur wins.
        if work > max_work:
            max_work = work
            best_idx = i
            best_min_dur = min_dur
        elif work == max_work:
            if min_dur < best_min_dur:
                best_idx = i
                best_min_dur = min_dur
    # Note: Floating point equality is risky. 
    # Better to use a tuple sort or a score function.
    
    # Let's use a score function for clarity and robustness.
    # Score = remaining_work. We want max.
    # Secondary key = -min_duration (we want min duration, so negative for max sort).
    
    scored_ops = []
    for i, op in enumerate(ready_operations):
        work = op['remaining_work']
        if op['machine_options']:
            min_dur = min(op['machine_options'].values())
        else:
            min_dur = float('inf')
        # To maximize work and minimize min_dur:
        # We can sort by (work, -min_dur)
        scored_ops.append((work, -min_dur, i))
        
    # Sort descending by work, then ascending by min_dur (via -min_dur)
    # Actually, if we sort descending, -min_dur will be larger for smaller min_dur.
    # Example: min_dur=5 -> -5. min_dur=3 -> -3. -3 > -5. So 3 comes first. Correct.
    
    scored_ops.sort(key=lambda x: (x[0], x[1]), reverse=True)
    
    best_idx = scored_ops[0][2]
    best_op = ready_operations[best_idx]
    
    # 2. Select Machine
    # Pick machine with minimum duration
    machine_options = best_op['machine_options']
    if not machine_options:
        return best_idx # Fallback if no machines (should not happen)
        
    # Find machine with min duration
    # Tie-breaker for machine: lower machine ID?
    best_machine = min(machine_options.keys(), key=lambda m: (machine_options[m], m))
    
    return (best_idx, best_machine)
