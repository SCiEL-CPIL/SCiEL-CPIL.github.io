"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def dispatch(ready_operations, time):
    """
    Look-ahead dispatch rule that simulates the immediate scheduling of each
    ready operation on its best machine, accounting for machine conflicts
    (blocking) of other ready operations. Selects the operation that minimizes
    the estimated global makespan.
    """
    if not ready_operations:
        return 0

    n = len(ready_operations)
    
    # Pre-compute the best (min duration) machine for each operation
    # and the remaining work for each job
    job_remaining = {}
    op_best_machine = [None] * n
    op_best_duration = [0] * n
    
    for i, op in enumerate(ready_operations):
        job_id = op["job"]
        if job_id not in job_remaining:
            job_remaining[job_id] = op["remaining_work"]
        else:
            if op["remaining_work"] > job_remaining[job_id]:
                job_remaining[job_id] = op["remaining_work"]
                
        machine_options = op["machine_options"]
        if machine_options:
            # Find machine with minimum duration
            best_m, best_d = min(machine_options.items(), key=lambda item: item[1])
            op_best_machine[i] = best_m
            op_best_duration[i] = best_d
        else:
            op_best_machine[i] = None
            op_best_duration[i] = 0

    best_index = 0
    best_makespan = float('inf')
    best_machine = None

    for i, op in enumerate(ready_operations):
        # We evaluate each machine option for this operation to find the one
        # that results in the lowest estimated makespan
        local_best_makespan = float('inf')
        local_best_machine = op_best_machine[i]
        
        machine_options = op["machine_options"]
        if not machine_options:
            # No machine options, skip or handle default
            continue

        for m, d in machine_options.items():
            # Simulate: Operation i starts now on machine m, duration d
            # Job i finishes at: time + remaining_work_i
            # (Assuming remaining_work includes current op duration)
            finish_i = time + op["remaining_work"]
            
            max_finish = finish_i
            
            # Check impact on other ready operations
            for k, other_op in enumerate(ready_operations):
                if k == i:
                    continue
                
                # If other operation k also wants machine m, it is blocked
                # until i finishes.
                # We assume k will use its best machine (min duration).
                # If k's best machine is m, it is delayed by d.
                # If k's best machine is not m, it is not delayed (it can use its best machine).
                
                # Note: This is a heuristic. In reality, k might choose a different machine
                # if m is blocked, but we assume it sticks to its best option or is flexible.
                # A more conservative approach: if m is blocked, k is delayed by d
                # only if k *must* use m or if m is k's best.
                # Given we want to minimize makespan, we assume k can adapt.
                # However, if k's best machine is m, and m is blocked, k's effective
                # duration increases or it waits.
                # Let's assume k waits for m if m is its best, otherwise it uses its next best?
                # To keep it simple and fast:
                # If op_best_machine[k] == m, then k is delayed by d.
                
                if op_best_machine[k] == m:
                    delay = d
                else:
                    delay = 0
                
                finish_k = time + other_op["remaining_work"] + delay
                if finish_k > max_finish:
                    max_finish = finish_k
            
            if max_finish < local_best_makespan:
                local_best_makespan = max_finish
                local_best_machine = m
        
        if local_best_makespan < best_makespan:
            best_makespan = local_best_makespan
            best_index = i
            best_machine = local_best_machine

    # Return index and the specific machine chosen for optimal performance
    # The contract allows (index, machine)
    return (int(best_index), best_machine)
# EVOLVE-BLOCK-END
