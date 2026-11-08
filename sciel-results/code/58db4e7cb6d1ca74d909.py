import time
import numpy as np

def dispatch(ready_operations, time):
    """
    Dispatches the next operation using a Critical Path / Longest Remaining Work heuristic.
    
    Strategy:
    - Instead of just picking the shortest operation (SPT), we estimate the "urgency" of each job.
    - We assume that jobs with more remaining work (sum of min durations of remaining ops) 
      are more critical to the makespan if they are not progressed.
    - We combine this with the processing time of the current operation on its best machine.
    - Specifically, we prioritize operations that belong to jobs with the highest 
      'remaining critical path length'. 
    - To approximate the critical path without a full graph search (which might be too slow 
      if called every dispatch step for many steps), we use the sum of minimum durations 
      of all remaining operations in the job as a proxy for the remaining workload.
    - We also consider the machine load. If a machine is heavily loaded (many ops queued), 
      we might prefer a different machine if available, but since we don't see the queue 
      directly, we stick to job-centric urgency.
    
    Score for an operation i:
    - primary: remaining_work_of_job[i] (higher is more urgent/critical)
    - secondary: min_duration_of_op[i] (lower is better, to keep machines busy with short tasks if urgency is similar)
    
    We want to MAXIMIZE urgency, so we minimize: -remaining_work + epsilon * min_duration
    Or simply: sort by remaining_work descending, then min_duration ascending.
    
    Let's use a weighted score to be robust:
    score = - (remaining_work / max_remaining_work) + (min_duration / max_min_duration) * small_weight
    Actually, simpler is often better. Let's try:
    Priority = remaining_work (of the job)
    Tie-breaker = min_duration of the op.
    
    Wait, the current SPT approach scores 0.82. The inspiration 1 (dynamic priority) scored 1.05.
    Inspiration 1 likely balanced SPT and remaining work.
    
    Let's try a "Longest Remaining Path" (LRP) heuristic.
    For each ready operation, we look at its job.
    We estimate the remaining makespan contribution of that job.
    A common heuristic in job-shop is to prioritize the job with the longest remaining 
    critical path. Since we don't have the full graph, we approximate the critical path 
    by the sum of the minimum processing times of all remaining operations in that job.
    
    Algorithm:
    1. For each ready op, calculate:
       a. min_dur: min duration across machines.
       b. rem_work: sum of min durations of ALL remaining ops in that job (including current).
       c. machine_load_factor: We don't know machine loads directly, but we can assume 
          that if a machine has many options, it might be less constrained? No.
    
    Let's stick to:
    1. Calculate 'job_urgency' = sum of min_durations of remaining ops in the job.
    2. Select the op from the job with the HIGHEST job_urgency.
    3. If multiple ops from different jobs have similar urgency, pick the one with 
       the shortest min_dur (SPT tie-breaker).
    4. For the selected op, pick the machine with the shortest duration.
    
    This is a "Critical Path" style heuristic.
    """
    if not ready_operations:
        return -1

    # Deadline guard
    start_time = time.time()
    deadline = 150.0  # Seconds
    safety_margin = 0.1
    
    # Pre-calculate metrics for each ready operation
    # We need to group by job to calculate remaining work per job
    job_remaining_work = {}
    op_min_dur = {}
    op_best_machine = {}
    
    # First pass: Calculate min duration and best machine for each ready op
    for i, op in enumerate(ready_operations):
        min_dur = float('inf')
        best_mach = -1
        for m, d in op['machine_options'].items():
            if d < min_dur:
                min_dur = d
                best_mach = m
        op_min_dur[i] = min_dur
        op_best_machine[i] = best_mach
        job_id = op['job']
        if job_id not in job_remaining_work:
            job_remaining_work[job_id] = 0.0
        
        # Note: 'remaining_ops' in the dict might be a list of future ops or a count.
        # The prompt says: "remaining_ops, remaining_work".
        # If 'remaining_work' is provided directly, use it. 
        # Let's check the dict keys. 
        # "machine_options {machine: duration}, remaining_ops, remaining_work"
        # So 'remaining_work' is likely the sum of durations for the rest of the job.
        # Does it include the current op? Usually "remaining" implies future.
        # Let's assume remaining_work is the work AFTER this op.
        # So total critical path approx = min_dur + remaining_work.
        
        job_remaining_work[job_id] += min_dur + op.get('remaining_work', 0)
        
        # Check deadline
        if time.time() > deadline - safety_margin:
            # Fallback to SPT if we are running out of time
            best_index = 0
            best_machine = op_best_machine[0]
            return (best_index, best_machine)

    # Find the job with the highest estimated remaining critical path
    # We want to prioritize jobs with MORE remaining work (critical path)
    # to ensure they are not left for last.
    
    max_work = max(job_remaining_work.values()) if job_remaining_work else 1.0
    
    best_index = -1
    best_machine = -1
    best_priority = -1.0
    best_spt = float('inf')
    
    for i, op in enumerate(ready_operations):
        job_id = op['job']
        work = job_remaining_work[job_id]
        
        # Priority: Higher work is better.
        # Normalize work to [0, 1]
        norm_work = work / max_work
        
        # SPT component: Lower min_dur is better.
        # We want to minimize: -norm_work + alpha * (min_dur / max_min_dur)
        # Or simply: Maximize norm_work. Tie-break with min_dur.
        
        # Let's use a composite score where lower is better.
        # Score = -norm_work + 0.1 * (op_min_dur[i] / max_min_dur_approx)
        # We don't have max_min_dur easily without another pass, but we can use a small constant 
        # or just use the rank.
        
        # Simpler: 
        # 1. Sort by work descending.
        # 2. Among those, pick lowest min_dur.
        
        # To do this in a single pass without sorting:
        # If work > best_work: update best.
        # If work == best_work and min_dur < best_min_dur: update best.
        
        # Floating point equality is risky. Let's use a small epsilon.
        # Or just use a weighted sum.
        # Weighted sum: Score = -work + 0.01 * min_dur
        # This heavily favors work.
        
        score = -work + 0.01 * op_min_dur[i]
        
        if score < best_priority:
            best_priority = score
            best_index = i
            best_machine = op_best_machine[i]
            
        # Check deadline
        if time.time() > deadline - safety_margin:
            if best_index != -1:
                return (best_index, best_machine)
            else:
                return (0, op_best_machine[0])

    if best_index == -1:
        return -1

    return (best_index, best_machine)
