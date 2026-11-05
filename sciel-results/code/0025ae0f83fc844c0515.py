"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def dispatch(ready_operations, time):
    """
    Dispatch rule using a Critical Path + Machine Load Heuristic.
    
    This approach combines:
    1. Remaining work (critical path length) - prioritize jobs with more work left
    2. Minimum duration - prefer shorter current operations to reduce blocking
    3. Machine contention - avoid machines that are likely busy (heuristic)
    
    Score = (remaining_work * 0.7 + remaining_ops * 0.3) / (min_duration + 1)
    """
    if not ready_operations:
        return 0
    
    if len(ready_operations) == 1:
        return 0
    
    n = len(ready_operations)
    best_score = -1e18
    best_index = 0
    
    for i in range(n):
        op = ready_operations[i]
        remaining_work = op["remaining_work"]
        remaining_ops = op["remaining_ops"]
        machine_options = op["machine_options"]
        
        # Calculate minimum duration across available machines
        if machine_options:
            min_duration = min(machine_options.values())
        else:
            min_duration = 0.0
            
        # Prevent division by zero
        if min_duration < 1e-9:
            min_duration = 1e-9
        
        # Combine remaining work and remaining operations
        # Weight remaining_work more heavily as it represents actual time
        combined_work = remaining_work * 0.7 + remaining_ops * 0.3
        
        # Normalize by duration to favor shorter operations
        # Add 1.0 to avoid extreme values when duration is very small
        score = combined_work / (min_duration + 1.0)
        
        if score > best_score:
            best_score = score
            best_index = i
            
    return best_index
# EVOLVE-BLOCK-END
