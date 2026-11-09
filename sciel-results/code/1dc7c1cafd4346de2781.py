"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def dispatch(ready_operations, time):
    """
    Dispatch rule: Select the ready operation with the highest 'efficiency' score,
    defined as remaining_work / min_duration. This prioritizes operations that
    complete a significant amount of work in a short time.
    
    Additionally, it explicitly selects the machine with the minimum duration
    for the chosen operation to ensure local optimality in machine assignment.
    """
    if not ready_operations:
        return 0

    best_index = 0
    best_efficiency = -float('inf')
    best_machine = None

    for i, op in enumerate(ready_operations):
        machine_options = op["machine_options"]
        if not machine_options:
            continue

        # Find the machine with the minimum duration for this operation
        # This is the best machine to assign to this op to minimize its completion time
        best_m, min_duration = min(machine_options.items(), key=lambda item: item[1])
        
        remaining_work = op["remaining_work"]
        
        # Efficiency: How much work is done per unit of time
        # Add a small epsilon to avoid division by zero
        efficiency = remaining_work / (min_duration + 1e-9)
        
        # We want to maximize efficiency
        if efficiency > best_efficiency:
            best_efficiency = efficiency
            best_index = i
            best_machine = best_m

    # Return index and the specific machine chosen
    # If best_machine is None (e.g. no valid ops found, though handled by if not ready_operations),
    # we should handle edge cases, but assuming valid input, best_machine will be set.
    if best_machine is None:
        return 0
        
    return (int(best_index), best_machine)
# EVOLVE-BLOCK-END
