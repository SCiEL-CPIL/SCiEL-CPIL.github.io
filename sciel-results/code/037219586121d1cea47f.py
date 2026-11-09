"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def dispatch(ready_operations, time):
    """
    Dispatch rule using Flexibility-Aware Critical Path heuristic.
    
    Strategy:
    1. Estimate urgency based on Remaining Work (Critical Path proxy).
    2. Adjust for immediate processing time (Shorter is better for throughput).
    3. Penalize high flexibility (Operations with many machine options can be scheduled later; 
       prioritize inflexible operations to avoid blocking).
    4. Select the specific machine that minimizes duration for the chosen operation.
    """
    if not ready_operations:
        return 0

    best_score = -1e9
    best_index = 0
    best_machine = None

    for i, op in enumerate(ready_operations):
        remaining_work = op["remaining_work"]
        machine_options = op["machine_options"]
        
        if not machine_options:
            # Fallback if no options (shouldn't happen in valid JSS)
            min_duration = remaining_work
            best_m_for_op = None
            flexibility = 1
        else:
            # Find the machine with the minimum duration
            best_m_for_op = min(machine_options, key=machine_options.get)
            min_duration = machine_options[best_m_for_op]
            flexibility = len(machine_options)
            
        # Heuristic:
        # Base score: Remaining Work / (Min Duration + 1)
        # This favors jobs with lots of work that can be started quickly.
        base_score = remaining_work / (min_duration + 1.0)
        
        # Flexibility penalty:
        # Divide by flexibility count. 
        # An operation with 5 machine options is 5x "easier" to place than one with 1.
        # So we lower its priority to save it for when we need to balance load.
        # Using 1/flexibility might be too strong. Let's use a softer penalty.
        # score = base_score / flexibility
        score = base_score / flexibility
        
        if score > best_score:
            best_score = score
            best_index = i
            best_machine = best_m_for_op

    # Return index and the optimal machine for that index
    if best_machine is not None:
        return (best_index, best_machine)
    else:
        return best_index
# EVOLVE-BLOCK-END
