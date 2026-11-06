"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def dispatch(ready_operations, time):
    """
    Dispatches the next ready operation.
    
    Strategy:
    1. Select the operation belonging to the job with the highest estimated
       remaining critical path length. This is approximated by `remaining_work`
       plus the minimum duration of the current operation.
    2. Assign the selected operation to the machine with the shortest
       processing time (SPT) to minimize the immediate blocking time.
    """
    if not ready_operations:
        return None

    best_index = -1
    best_score = -float('inf')
    best_machine = None

    for i, op in enumerate(ready_operations):
        machine_options = op["machine_options"]
        if not machine_options:
            continue
            
        # Calculate minimum duration for the current operation
        min_duration = min(machine_options.values())
        
        # Estimated total remaining critical path
        # We add min_duration to remaining_work to ensure the current step's
        # cost is heavily weighted in the urgency calculation.
        score = op["remaining_work"] + min_duration
        
        # Tie-breaker: Prefer jobs with more remaining operations (longer chains)
        # This helps in cases where remaining_work is similar but chain length differs.
        # A small epsilon ensures score dominates.
        tie_breaker = op["remaining_ops"]
        
        # Composite score
        final_score = score + 1e-9 * tie_breaker
        
        if final_score > best_score:
            best_score = final_score
            best_index = i
            
            # Select machine with minimum duration
            # If multiple machines have the same min duration, pick the first one found
            for m, d in machine_options.items():
                if d == min_duration:
                    best_machine = m
                    break
                    
    if best_index == -1:
        # Fallback to first operation if something went wrong
        return 0

    # Return index and explicitly selected machine
    return (best_index, best_machine)
# EVOLVE-BLOCK-END
