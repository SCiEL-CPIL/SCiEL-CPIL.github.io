"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    # For each ready operation, compute:
    # - remaining_work: total remaining processing time for the job
    # - min_duration: shortest processing time across available machines
    # Select operation with max remaining_work, tie-break by min min_duration (SPT)
    # Then assign to the machine with the shortest duration for that operation.

    scores = []
    min_durations = []
    best_machines = []
    for op in ready_operations:
        mo = op["machine_options"]
        if mo:
            min_dur = min(mo.values())
            best_machine = min(mo, key=mo.get)
        else:
            min_dur = float('inf')
            best_machine = None
        min_durations.append(min_dur)
        best_machines.append(best_machine)
        # Primary: remaining_work (higher is better)
        # Secondary: min_duration (lower is better) -> use negative for argmax
        scores.append((op["remaining_work"], -min_dur))

    # Find the best index by tuple comparison (max remaining_work, then min min_duration)
    best_idx = 0
    best_score = scores[0]
    for i in range(1, len(scores)):
        if scores[i] > best_score:
            best_score = scores[i]
            best_idx = i

    # Return (index, machine) to also specify machine assignment
    if best_machines[best_idx] is not None:
        return (best_idx, best_machines[best_idx])
    return best_idx
# EVOLVE-BLOCK-END
