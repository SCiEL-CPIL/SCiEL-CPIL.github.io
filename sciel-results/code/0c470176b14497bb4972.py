"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def dispatch(ready_operations, time):
    """
    Dispatch rule for flexible job-shop scheduling.
    
    Uses a Min-Processing-Time (MPT) approach with Load Balancing:
    1. Identify the eligible (operation, machine) pair with the minimum processing time (duration).
    2. If there are ties in minimum duration, select the pair on the machine with the lowest 
       potential contention (fewest other ready operations that can use that machine).
    3. If contention is also tied, select the operation with the highest remaining_work 
       (urgency/critical path heuristic).
    
    Returns: (index, machine) tuple.
    """
    if not ready_operations:
        return 0
    
    n = len(ready_operations)
    
    # Pre-calculate contention for each machine to use in tie-breaking
    # contention[m] = count of ready ops that can use machine m
    contention = {}
    for op in ready_operations:
        for m in op["machine_options"].keys():
            contention[m] = contention.get(m, 0) + 1

    best_duration = float('inf')
    best_idx = 0
    best_machine = None
    best_contention = float('inf')
    best_rw = -1.0

    for i, op in enumerate(ready_operations):
        rw = op["remaining_work"]
        if rw <= 0:
            continue
            
        for m, d in op["machine_options"].items():
            if d <= 0:
                continue
            
            # Primary criterion: Minimum Duration
            if d < best_duration:
                best_duration = d
                best_idx = i
                best_machine = m
                best_contention = contention.get(m, 0)
                best_rw = rw
            elif d == best_duration:
                # Secondary criterion: Lowest Contention on the machine
                c = contention.get(m, 0)
                if c < best_contention:
                    best_contention = c
                    best_idx = i
                    best_machine = m
                    best_rw = rw
                elif c == best_contention:
                    # Tertiary criterion: Highest Remaining Work (Urgency)
                    if rw > best_rw:
                        best_rw = rw
                        best_idx = i
                        best_machine = m

    # Fallback if no valid pair found
    if best_machine is None:
        if ready_operations:
            best_idx = 0
            machine_options = ready_operations[0]["machine_options"]
            if machine_options:
                best_machine = list(machine_options.keys())[0]
            else:
                best_machine = 0
        else:
            return 0
            
    return (best_idx, best_machine)
# EVOLVE-BLOCK-END
