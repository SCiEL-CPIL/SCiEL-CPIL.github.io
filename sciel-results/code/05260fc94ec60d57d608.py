"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as time_module

def dispatch(ready_operations, time):
    """
    Dispatch rule for Open-Shop Scheduling.
    Returns (index, machine) to maximize a composite score of job priority and operation efficiency.
    """
    if not ready_operations:
        return -1

    best_score = -float('inf')
    best_index = 0
    best_machine = None

    for i, op in enumerate(ready_operations):
        machine_options = op.get("machine_options", {})
        if not machine_options:
            continue

        # Find the machine with the minimum processing time for this operation
        min_dur = float('inf')
        best_m = None
        for m, d in machine_options.items():
            if d < min_dur:
                min_dur = d
                best_m = m

        if best_m is None:
            continue

        # Retrieve job metrics
        remaining_work = op.get("remaining_work", 0)
        remaining_ops = op.get("remaining_ops", 0)

        # Composite Score:
        # 1. remaining_work: Primary factor to minimize job tail (makespan).
        # 2. remaining_ops: Secondary factor to prioritize jobs with deeper queues.
        # 3. Inverse duration: Tertiary factor to favor operations that complete quickly,
        #    freeing up the machine sooner (smoothed to avoid division by zero).
        
        # Weights tuned to prioritize job completion while balancing machine utilization.
        score = 1.0 * remaining_work + 0.2 * remaining_ops + 2.0 / (min_dur + 1.0)

        if score > best_score:
            best_score = score
            best_index = i
            best_machine = best_m

    # Return the best operation index and its optimal machine
    return (best_index, best_machine)
# EVOLVE-BLOCK-END
