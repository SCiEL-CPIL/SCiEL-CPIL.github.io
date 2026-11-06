"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def dispatch(ready_operations, time):
    """
    Dispatch rule: Select the ready operation using a hybrid of Most Work Remaining (MWR)
    and Least Flexible First (LFF).
    
    Heuristic:
    1. MWR: Prioritize operations from jobs with more remaining work, as they are
       more likely to be on the critical path.
    2. LFF: Prioritize operations with fewer machine options. Operations with less
       flexibility should be scheduled earlier to avoid being blocked by machine
       contention later in the schedule.
    
    Score = remaining_work / num_machine_options
    """
    if not ready_operations:
        return 0
    
    n = len(ready_operations)
    
    scores = []
    for op in ready_operations:
        machine_options = op["machine_options"]
        remaining_work = op["remaining_work"]
        
        num_options = len(machine_options)
        if num_options == 0:
            num_options = 1 # Avoid division by zero, treat as high priority/fixed
            
        # Higher score for more work, higher score for less flexibility (fewer options)
        # This effectively weights the remaining work by the inverse of flexibility
        score = remaining_work / float(num_options)
        
        scores.append(score)
    
    # Select the operation with the highest score
    best_idx = int(np.argmax(scores))
    
    return best_idx
# EVOLVE-BLOCK-END
