"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def dispatch(ready_operations, time):
    """
    Dispatch rule for flexible job-shop scheduling.
    
    Uses a Contention-Aware Critical Path heuristic:
    1. For each operation, determine the 'natural' machine (minimum processing time).
    2. Calculate machine contention based on how many operations prefer that machine.
    3. Estimate the effective processing time for each operation on each eligible machine,
       penalizing machines with high contention.
    4. Score each operation by its remaining work divided by its minimum estimated effective processing time.
    5. Select the operation with the highest score.
    6. For the selected operation, choose the machine that minimizes the effective processing time.
    
    Returns: (index, machine) tuple.
    """
    if not ready_operations:
        return 0
    
    n = len(ready_operations)
    
    # Step 1: Determine 'natural' machine (min duration) for each op and min duration
    natural_machines = [None] * n
    min_durations = np.zeros(n)
    
    for i, op in enumerate(ready_operations):
        machine_options = op["machine_options"]
        if not machine_options:
            natural_machines[i] = -1
            min_durations[i] = float('inf')
            continue
        
        best_m = None
        min_d = float('inf')
        for m, d in machine_options.items():
            if d < min_d:
                min_d = d
                best_m = m
        natural_machines[i] = best_m
        min_durations[i] = min_d
        
    # Step 2: Calculate machine contention
    # Count how many ops have each machine as their 'natural' (min duration) machine
    machine_counts = {}
    for m in natural_machines:
        if m is not None and m != -1:
            machine_counts[m] = machine_counts.get(m, 0) + 1
            
    # Step 3: Calculate effective processing time for each op on each machine
    # effective_time(i, m) = duration(i, m) * (1 + contention[m])
    # We want the minimum effective time for each op
    min_effective_times = np.zeros(n)
    best_machines_for_op = [None] * n
    
    for i, op in enumerate(ready_operations):
        machine_options = op["machine_options"]
        if not machine_options:
            min_effective_times[i] = float('inf')
            continue
            
        min_eff = float('inf')
        best_m = None
        for m, d in machine_options.items():
            contention = machine_counts.get(m, 0)
            # Penalize crowded machines. 
            # Using (1 + contention) means if 5 ops want machine 0, 
            # the time is effectively multiplied by 6.
            eff_time = d * (1 + contention)
            if eff_time < min_eff:
                min_eff = eff_time
                best_m = m
                
        min_effective_times[i] = min_eff
        best_machines_for_op[i] = best_m
        
    # Step 4: Score each operation
    # Score = remaining_work / (1 + min_effective_time)
    # This prioritizes high remaining work (Critical Path) but penalizes long effective processing times.
    scores = np.zeros(n)
    for i, op in enumerate(ready_operations):
        rw = op["remaining_work"]
        eff_t = min_effective_times[i]
        
        if eff_t == float('inf'):
            scores[i] = -np.inf
        else:
            # Add 1 to denominator to avoid division by zero and smooth the scale
            scores[i] = rw / (1 + eff_t)
            
    # Step 5: Select best operation
    best_idx = int(np.argmax(scores))
    
    # Step 6: Select best machine for the chosen operation
    chosen_machine = best_machines_for_op[best_idx]
    
    # Fallback if something went wrong
    if chosen_machine is None:
        # Pick any available machine
        machine_options = ready_operations[best_idx]["machine_options"]
        if machine_options:
            chosen_machine = list(machine_options.keys())[0]
        else:
            chosen_machine = 0
            
    return (best_idx, chosen_machine)
# EVOLVE-BLOCK-END
