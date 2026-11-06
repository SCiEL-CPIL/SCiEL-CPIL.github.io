"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import defaultdict


def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    # Step 1: Estimate the "load" or "pressure" on each machine based on the current ready set.
    # We use the sum of durations of all ready operations that can be assigned to that machine.
    # This serves as a proxy for how "busy" or "contended" a machine is in the immediate future.
    machine_work = defaultdict(float)
    for op in ready_operations:
        for machine, duration in op["machine_options"].items():
            machine_work[machine] += duration

    # Step 2: For each ready operation, simulate its assignment to the least-loaded compatible machine
    # and calculate the resulting maximum machine load (L-infinity norm).
    # We want to pick the operation that results in the smallest increase to the global max load.
    
    best_index = 0
    best_score = float('inf')
    
    # Pre-compute the max work for all machines except the one being updated, to speed up the max calculation?
    # Since N is small (10 jobs, 5 machines, ready ops <= 10), O(N^2) is fine.
    
    max_works = list(machine_work.values())
    global_max_work = max(max_works) if max_works else 0
    
    for i, op in enumerate(ready_operations):
        # Find the machine with the minimum current estimated work among options
        # Tie-breaker: if work is equal, pick the one with smaller duration for this op?
        # The water filling logic says: assign to least loaded.
        min_work_machine = None
        min_work_val = float('inf')
        
        for machine, duration in op["machine_options"].items():
            w = machine_work.get(machine, 0)
            if w < min_work_val:
                min_work_val = w
                min_work_machine = machine
            elif w == min_work_val:
                # Tie-breaker: prefer the machine where this specific op has shorter duration?
                # Or just keep the first one found.
                pass

        if min_work_machine is None:
            continue

        duration_on_min_machine = op["machine_options"][min_work_machine]
        new_load_on_min_machine = min_work_val + duration_on_min_machine
        
        # The new global max load is the max of:
        # 1. The new load on the machine we just assigned to
        # 2. The max load of all OTHER machines (which remain unchanged)
        
        # To get max of other machines, we can just take the global max, 
        # unless the machine we updated WAS the one holding the global max.
        
        # Let's calculate max of other machines explicitly to be safe and simple.
        max_other_load = 0
        for machine, work in machine_work.items():
            if machine != min_work_machine:
                if work > max_other_load:
                    max_other_load = work
        
        resulting_max_load = max(new_load_on_min_machine, max_other_load)
        
        # We want to minimize this resulting max load
        if resulting_max_load < best_score:
            best_score = resulting_max_load
            best_index = i

    return int(best_index)
# EVOLVE-BLOCK-END
