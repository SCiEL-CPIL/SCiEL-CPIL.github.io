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
    
    # Pre-compute max remaining work excluding each job to speed up evaluation
    # We'll use a two-max approach for efficiency
    if job_remaining:
        items = list(job_remaining.items())
        if len(items) == 1:
            max_other_map = {items[0][0]: 0}
        else:
            # Find top two max values
            sorted_items = sorted(items, key=lambda x: x[1], reverse=True)
            top1_job, top1_val = sorted_items[0]
            top2_val = sorted_items[1][1] if len(sorted_items) > 1 else 0
            max_other_map = {}
            for jid, rw in items:
                if jid == top1_job:
                    max_other_map[jid] = top2_val
                else:
                    max_other_map[jid] = top1_val
    else:
        max_other_map = {}
    
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
        old_remaining = job_remaining.get(job_id, 0)
        new_remaining = max(0, old_remaining - min_duration)
        
        # Calculate the new critical path using pre-computed max_other
        max_other = max_other_map.get(job_id, 0)
        new_critical_path = max(max_other, new_remaining)
        
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
