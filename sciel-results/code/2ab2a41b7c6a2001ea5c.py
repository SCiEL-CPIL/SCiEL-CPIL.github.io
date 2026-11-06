"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import defaultdict


def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    # Step 1: Estimate the "remaining work" on each machine based on the current ready set.
    # This represents the total duration of all ready operations that must be processed on that machine.
    machine_work = defaultdict(float)
    for op in ready_operations:
        for machine, duration in op["machine_options"].items():
            machine_work[machine] += duration

    # Step 2: For each ready operation, simulate its assignment to the least-loaded compatible machine
    # and calculate the resulting maximum machine load (L-infinity norm) of the *remaining* work.
    # We want to pick the operation that results in the smallest maximum remaining load.
    
    best_index = 0
    best_score = float('inf')
    
    for i, op in enumerate(ready_operations):
        # Find the machine with the minimum current estimated work among options
        min_work_machine = None
        min_work_val = float('inf')
        
        for machine, duration in op["machine_options"].items():
            w = machine_work.get(machine, 0)
            if w < min_work_val:
                min_work_val = w
                min_work_machine = machine
            elif w == min_work_val:
                # Tie-breaker: If loads are equal, prefer the machine where this specific op has
                # a LARGER duration. This reduces the remaining load on that machine more,
                # which is beneficial if that machine is part of the bottleneck.
                if min_work_machine is not None:
                    prev_dur = op["machine_options"][min_work_machine]
                    if duration > prev_dur:
                        min_work_machine = machine

        if min_work_machine is None:
            continue

        duration_on_min_machine = op["machine_options"][min_work_machine]
        # The new remaining load on the assigned machine is the old load minus the duration of the started op.
        new_load_on_min_machine = min_work_val - duration_on_min_machine
        
        # The new global max load is the max of:
        # 1. The new load on the machine we just updated
        # 2. The max load of all OTHER machines (which remain unchanged)
        
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
