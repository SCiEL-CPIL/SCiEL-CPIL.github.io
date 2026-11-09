"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    # Combined heuristic: prioritize by a weighted combination of
    # remaining work and urgency (fewer remaining operations = more urgent)
    # Also consider the shortest processing time among machine options
    # to reduce potential waiting times.
    
    best_idx = 0
    best_score = -float('inf')
    
    for i, op in enumerate(ready_operations):
        # Get remaining work
        work = op.get("remaining_work", 0)
        if work == 0:
            options = op.get("machine_options", {})
            if options:
                work = min(options.values())
            else:
                work = 0
        
        # Get number of remaining operations
        remaining_ops = op.get("remaining_ops", 1)
        if remaining_ops <= 0:
            remaining_ops = 1
        
        # Get minimum processing time among machine options
        options = op.get("machine_options", {})
        if options:
            min_duration = min(options.values())
        else:
            min_duration = work
        
        # Score: balance remaining work, urgency, and short processing time
        # Higher remaining work = higher priority
        # Fewer remaining ops = higher urgency (closer to completion)
        # Shorter min duration = less blocking
        
        # Normalize and combine
        work_score = work / (work + 1.0)  # Bounded 0-1
        urgency_score = 1.0 / remaining_ops  # Higher when fewer ops remain
        duration_score = 1.0 / (min_duration + 1.0)  # Higher for shorter durations
        
        # Weighted combination
        score = 0.5 * work_score + 0.3 * urgency_score + 0.2 * duration_score
        
        if score > best_score:
            best_score = score
            best_idx = i
            
    return best_idx
# EVOLVE-BLOCK-END
