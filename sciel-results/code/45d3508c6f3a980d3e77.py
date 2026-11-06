"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def dispatch(ready_operations, time):
    """
    Dispatch rule for flexible job-shop scheduling.
    
    Uses a Workload-Aware Greedy Heuristic:
    1. Calculate the estimated total workload for each machine based on ready operations.
    2. For each ready operation and each eligible machine, calculate a score:
       Score = remaining_work / (duration + estimated_workload_on_machine)
    3. This prioritizes operations with high remaining work (urgency) that can be
       processed quickly on machines with low estimated backlog.
    4. Select the operation-machine pair with the highest score.
    
    Returns: (index, machine) tuple.
    """
    if not ready_operations:
        return 0
    
    n = len(ready_operations)
    
    # Step 1: Calculate estimated workload for each machine
    # estimated_workload[m] = sum of durations of all ready ops that can use machine m
    estimated_workload = {}
    for op in ready_operations:
        for m, d in op["machine_options"].items():
            estimated_workload[m] = estimated_workload.get(m, 0) + d
            
    # Step 2: Evaluate all (operation, machine) pairs
    best_score = -1.0
    best_idx = 0
    best_machine = None
    
    for i, op in enumerate(ready_operations):
        rw = op["remaining_work"]
        if rw <= 0:
            continue
            
        for m, d in op["machine_options"].items():
            if d <= 0:
                continue
                
            workload = estimated_workload.get(m, 0)
            # Effective cost: duration plus the estimated backlog on that machine
            # We add a small epsilon to avoid division by zero if workload is 0, 
            # but d > 0 is guaranteed.
            eff_cost = d + workload
            
            # Score: Urgency / Estimated Total Time on Machine
            score = rw / eff_cost
            
            if score > best_score:
                best_score = score
                best_idx = i
                best_machine = m
                
    # Fallback if no valid pair found
    if best_machine is None:
        # Pick the first operation and its first machine
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
