"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def dispatch(ready_operations, time):
    """
    Look-ahead dispatch rule that simulates the immediate completion of each
    ready operation to calculate the resulting reduction in the global
    critical path (maximum remaining work), accounting for machine conflicts.
    Selects the operation and machine that minimizes the estimated maximum
    job completion time.
    """
    if not ready_operations:
        return 0

    n = len(ready_operations)
    
    # Pre-compute properties for each ready operation
    op_min_duration = [0] * n
    op_best_machine = [None] * n
    op_remaining_work = [0] * n
    op_job_id = [None] * n
    op_machine_options = [None] * n
    
    for i, op in enumerate(ready_operations):
        op_job_id[i] = op["job"]
        op_remaining_work[i] = op["remaining_work"]
        machine_options = op["machine_options"]
        op_machine_options[i] = machine_options
        
        if machine_options:
            # Find machine with minimum duration
            best_m, best_d = min(machine_options.items(), key=lambda item: item[1])
            op_best_machine[i] = best_m
            op_min_duration[i] = best_d
        else:
            op_best_machine[i] = None
            op_min_duration[i] = 0

    best_index = 0
    best_makespan = float('inf')
    best_machine = None

    for i, op in enumerate(ready_operations):
        local_best_makespan = float('inf')
        local_best_machine = op_best_machine[i]
        
        machine_options = op["machine_options"]
        if not machine_options:
            continue

        for m, d in machine_options.items():
            # Estimate finish time for job i if started on machine m
            # remaining_work is assumed to be the sum of min durations.
            # If we pick duration d, the extra time is d - min_duration.
            # Finish time = time + remaining_work + (d - min_duration)
            extra_time_i = d - op_min_duration[i]
            finish_i = time + op_remaining_work[i] + extra_time_i
            
            max_finish = finish_i
            
            # Check impact on other ready operations
            for k, other_op in enumerate(ready_operations):
                if k == i:
                    continue
                
                # If other operation k's best machine is m, it is blocked
                # and delayed by d.
                # We assume if it's not the best machine, it can use an alternative
                # without delay (heuristic).
                
                if op_best_machine[k] == m:
                    # Delayed by the full duration d of operation i
                    delay = d
                else:
                    delay = 0
                
                # Finish time for job k
                # If delayed, it starts later.
                # Its base finish time is time + remaining_work[k]
                # Plus the delay.
                finish_k = time + op_remaining_work[k] + delay
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
    return (int(best_index), best_machine)
# EVOLVE-BLOCK-END
