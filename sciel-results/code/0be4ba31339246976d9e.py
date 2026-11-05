"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def dispatch(ready_operations, time):
    """
    Look-ahead dispatch rule: simulate the immediate completion of each ready
    operation and select the one that yields the greatest decrease in the
    global critical path (maximum remaining work across all jobs).
    
    If there is no clear decrease, fall back to a tie-breaking heuristic
    that prefers operations with more remaining work on their job or fewer
    machine options (tighter constraints).
    """
    if not ready_operations:
        return 0
    
    n = len(ready_operations)
    
    # Pre-compute current remaining work for each job involved
    # and the current global critical path
    job_remaining = {}
    for op in ready_operations:
        job_id = op["job"]
        if job_id not in job_remaining:
            job_remaining[job_id] = op["remaining_work"]
        else:
            # remaining_work should be consistent per job; take max to be safe
            if op["remaining_work"] > job_remaining[job_id]:
                job_remaining[job_id] = op["remaining_work"]
    
    current_critical_path = max(job_remaining.values()) if job_remaining else 0
    
    best_index = 0
    best_score = -float('inf')
    
    for i, op in enumerate(ready_operations):
        job_id = op["job"]
        machine_options = op["machine_options"]
        
        # Find the minimum duration option for this operation
        if machine_options:
            min_duration = min(machine_options.values())
        else:
            min_duration = 0
        
        # Simulate completion: remaining work for this job decreases by min_duration
        # (We use min_duration as a proxy for the actual duration, since we don't
        # know which machine will be assigned, but this gives us a lower bound
        # on the reduction)
        old_remaining = job_remaining.get(job_id, 0)
        new_remaining = max(0, old_remaining - min_duration)
        
        # Calculate the new critical path
        # We need to check all jobs, but only this job's remaining work changes
        if job_id == job_id:  # always true, just for clarity
            # Find max of all other jobs and this job's new remaining
            max_other = 0
            for jid, rw in job_remaining.items():
                if jid != job_id:
                    if rw > max_other:
                        max_other = rw
            new_critical_path = max(max_other, new_remaining)
        else:
            new_critical_path = current_critical_path
        
        # The reduction in critical path
        reduction = current_critical_path - new_critical_path
        
        # Secondary tie-breakers to make the rule more robust:
        # 1. Prefer operations with more remaining work (more urgent)
        # 2. Prefer operations with fewer machine options (tighter constraints)
        # 3. Prefer operations with fewer remaining ops (closer to completion)
        
        remaining_ops = op.get("remaining_ops", 1)
        num_machine_options = len(machine_options) if machine_options else 1
        
        # Combine primary and secondary scores
        # Primary: reduction in critical path (higher is better)
        # Secondary: use small weights to break ties
        secondary_score = (
            op["remaining_work"] * 0.001 -  # more remaining work = slightly higher priority
            num_machine_options * 0.0005 -  # fewer options = slightly higher priority
            remaining_ops * 0.0001           # fewer remaining ops = slightly higher priority
        )
        
        total_score = reduction + secondary_score
        
        if total_score > best_score:
            best_score = total_score
            best_index = i
    
    return int(best_index)
# EVOLVE-BLOCK-END
