"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def dispatch(ready_operations, time):
    if not ready_operations:
        return 0 # Should not happen if called correctly, but safe

    # Calculate the minimum processing time for each operation
    # assuming the simulator will pick the fastest machine.
    # If the simulator picks differently, this is still a good proxy for "weight".
    min_durations = []
    for op in ready_operations:
        options = op["machine_options"]
        if options:
            min_durations.append(min(options.values()))
        else:
            min_durations.append(0)
            
    # Longest Processing Time (LPT) heuristic
    # Prioritize the operation with the longest minimum duration.
    # This helps in clearing bottleneck operations early.
    idx = int(np.argmax(min_durations))
    return idx
# EVOLVE-BLOCK-END
