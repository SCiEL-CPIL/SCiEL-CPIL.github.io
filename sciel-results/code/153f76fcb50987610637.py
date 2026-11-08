import numpy as np
import time as _time

# Deadline guard: ensure we return well before 150s wall-clock limit
_START_TIME = _time.time()
_DEADLINE = _START_TIME + 140.0  # 10s safety margin

def dispatch(ready_operations, time):
    if not ready_operations:
        return 0
    
    # Check deadline
    if _time.time() > _DEADLINE:
        return 0
    
    best_idx = 0
    best_score = float('-inf')
    best_machine = None
    
    n = len(ready_operations)
    
    for i, op in enumerate(ready_operations):
        options = op['machine_options']
        if not options:
            continue
        
        # Find the machine with minimum duration for this operation
        min_dur = min(options.values())
        best_machine_for_op = next(m for m, d in options.items() if d == min_dur)
        
        remaining_work = op['remaining_work']
        remaining_ops = op['remaining_ops']
        
        # Multi-criteria heuristic:
        # 1. WSPT-like: remaining_work / processing_time (long jobs first)
        # 2. Critical path: prefer operations with more remaining work
        # 3. Machine load balancing: prefer machines with less current load
        #    (approximated by preferring shorter durations)
        # 4. Urgency: operations with fewer remaining ops might be closer to completion
        
        # Primary: WSPT ratio
        wspt_score = remaining_work / (min_dur + 1e-9)
        
        # Secondary: critical path weight (more remaining work = more critical)
        cp_weight = remaining_work
        
        # Tertiary: prefer shorter processing times (reduces makespan)
        dur_penalty = min_dur
        
        # Combine: weighted sum
        # Higher WSPT is better, higher CP is better, lower duration is better
        score = wspt_score * 1.0 + cp_weight * 0.1 - dur_penalty * 0.01
        
        # Bonus for operations that are "almost done" (few remaining ops)
        if remaining_ops > 0:
            urgency_bonus = 1.0 / (remaining_ops + 1)
            score += urgency_bonus * 0.5
        
        if score > best_score:
            best_score = score
            best_idx = i
            best_machine = best_machine_for_op
    
    return (best_idx, best_machine)
