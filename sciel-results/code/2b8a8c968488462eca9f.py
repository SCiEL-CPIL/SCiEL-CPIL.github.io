"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as time_module

def dispatch(ready_operations, time):
    """
    Dispatches the next operation to run.
    
    Strategy:
    1. For each ready operation, determine the "best" machine as the one with the 
       shortest processing time (duration). This is the SPT machine selection rule.
    2. Calculate a priority score for each operation based on a work-to-time ratio:
       score = remaining_work / (shortest_duration + epsilon)
       This prioritizes operations that have significant remaining work but can be 
       completed quickly on their best machine.
    3. Select the operation with the highest score.
    4. Return the index of the selected operation and the index of its best machine.
    """
    if not ready_operations:
        return 0 # Fallback, though usually shouldn't be empty if called

    best_index = -1
    best_score = -1
    best_machine = -1
    
    epsilon = 1e-9
    
    for i, op in enumerate(ready_operations):
        machine_options = op["machine_options"]
        remaining_work = op["remaining_work"]
        
        # Find the machine with the shortest processing time (SPT)
        min_duration = float('inf')
        best_m = -1
        
        for m, dur in machine_options.items():
            if dur < min_duration:
                min_duration = dur
                best_m = m
        
        # If no machine found (should not happen), skip or fallback
        if best_m == -1:
            continue
            
        # Calculate work-to-time ratio score
        # We use the shortest duration as the "time" component
        # Higher remaining work and lower duration => Higher score
        score = remaining_work / (min_duration + epsilon)
        
        if score > best_score:
            best_score = score
            best_index = i
            best_machine = best_m

    # Fallback if something went wrong
    if best_index == -1:
        return 0

    # Return tuple (index, machine) to explicitly assign the machine
    return (best_index, best_machine)
# EVOLVE-BLOCK-END
