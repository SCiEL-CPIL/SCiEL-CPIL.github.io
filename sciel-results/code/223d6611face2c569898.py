"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

# Global start time for deadline guarding
_start_time = time.time()
_TIME_LIMIT = 145.0  # 5 seconds margin from 150s limit

def dispatch(ready_operations, time):
    """
    Picks the ready operation to start next.
    Returns a tuple (index, machine_id) to explicitly select the machine.
    Heuristic: Maximize (remaining_work / min_duration) to balance job urgency and machine efficiency.
    """
    # Deadline guard
    if time.time() - _start_time > _TIME_LIMIT:
        # Fallback to simple MWR or first available if time is up
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))

    if not ready_operations:
        return 0  # Should not happen if called only when ready ops exist, but safe default

    best_score = -1.0
    best_index = 0
    best_machine = None

    for i, op in enumerate(ready_operations):
        machine_options = op["machine_options"]
        if not machine_options:
            continue
            
        # Find the machine with the shortest duration for this operation
        min_duration = float('inf')
        min_machine = None
        for m, d in machine_options.items():
            if d < min_duration:
                min_duration = d
                min_machine = m
        
        if min_machine is None:
            continue

        remaining_work = op["remaining_work"]
        
        # Heuristic: Efficiency = Work Remaining / Time to Process
        # This prioritizes jobs with lots of work that can be processed quickly,
        # or jobs with little work that are very fast (clearing them out).
        # Avoid division by zero
        if min_duration > 0:
            score = remaining_work / min_duration
        else:
            score = remaining_work
            
        if score > best_score:
            best_score = score
            best_index = i
            best_machine = min_machine

    # Return index and the selected machine
    return (best_index, best_machine)
# EVOLVE-BLOCK-END
