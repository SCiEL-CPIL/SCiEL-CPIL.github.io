"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def dispatch(ready_operations, time):
    """
    Look-ahead dispatch rule that simulates the immediate completion of each
    ready operation to calculate the resulting reduction in the global critical
    path (maximum remaining work across all jobs). Selects the operation that
    yields the greatest decrease in this makespan lower bound.
    """
    if not ready_operations:
        return 0

    n = len(ready_operations)
    
    # Calculate the current global critical path (max remaining work across jobs)
    # This represents the earliest any job can finish, ignoring machine conflicts
    job_remaining = {}
    for i, op in enumerate(ready_operations):
        job_id = op["job"]
        if job_id not in job_remaining:
            job_remaining[job_id] = op["remaining_work"]
        else:
            if op["remaining_work"] > job_remaining[job_id]:
                job_remaining[job_id] = op["remaining_work"]
    
    # The current makespan lower bound is the max of all job remaining works
    current_critical_path = max(job_remaining.values()) if job_remaining else 0
    
    best_index = 0
    best_improvement = float('-inf')
    best_machine = None

    for i, op in enumerate(ready_operations):
        job_id = op["job"]
        machine_options = op["machine_options"]
        
        if not machine_options:
            continue
            
        # For this operation, simulate completing it immediately
        # The operation's job remaining work decreases by the duration of the current op
        # We need to find the duration of the current operation on each machine
        
        local_best_improvement = float('-inf')
        local_best_machine = None
        
        for m, d in machine_options.items():
            # If we schedule op i on machine m with duration d:
            # The job's remaining work decreases by d (since we're starting this op)
            # Wait: remaining_work usually includes the current op.
            # If we start the op, the "remaining" after completion would be remaining_work - d?
            # Actually, in a job-shop, remaining_work is the total work left including current op.
            # If we start the op now, it takes 'd' time. The job's new remaining work 
            # (after this op completes) would be remaining_work - d.
            # But the makespan lower bound is the max remaining work.
            # If we complete this op, the job's remaining work drops from remaining_work to remaining_work - d.
            
            # However, the "critical path" is the max over all jobs.
            # If this job was the one with the max remaining work, and we reduce it,
            # the new critical path might be lower.
            
            # Let's calculate the new critical path if we complete op i on machine m.
            # New remaining work for job i: remaining_work - d
            # Other jobs' remaining work stays the same.
            
            new_job_remaining = op["remaining_work"] - d
            if new_job_remaining < 0:
                new_job_remaining = 0
            
            # Calculate new critical path
            max_new_remaining = 0
            for j_id, rem in job_remaining.items():
                if j_id == job_id:
                    if new_job_remaining > max_new_remaining:
                        max_new_remaining = new_job_remaining
                else:
                    if rem > max_new_remaining:
                        max_new_remaining = rem
            
            # The improvement is the reduction in the critical path
            improvement = current_critical_path - max_new_remaining
            
            if improvement > local_best_improvement:
                local_best_improvement = improvement
                local_best_machine = m
        
        if local_best_improvement > best_improvement:
            best_improvement = local_best_improvement
            best_index = i
            best_machine = local_best_machine

    # Fallback if no improvement found (shouldn't happen with valid inputs)
    if best_machine is None:
        # Pick the operation with the longest remaining work on its best machine
        best_index = 0
        best_remaining = -1
        for i, op in enumerate(ready_operations):
            if op["machine_options"]:
                min_d = min(op["machine_options"].values())
                if op["remaining_work"] > best_remaining:
                    best_remaining = op["remaining_work"]
                    best_index = i
                    best_machine = min(op["machine_options"].items(), key=lambda x: x[1])[0]
        if best_machine is None and ready_operations:
            best_machine = list(ready_operations[0]["machine_options"].keys())[0] if ready_operations[0]["machine_options"] else 0

    return (int(best_index), best_machine)
# EVOLVE-BLOCK-END
