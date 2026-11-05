"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

# Global deadline guard
_START_TIME = _time.time()
_DEADLINE = 140.0  # 10s safety margin before 150s

def dispatch(ready_operations, time):
    # Check deadline
    if _time.time() > _DEADLINE:
        # Fallback: Return first index to prevent crash/timeout score 0
        return 0

    if not ready_operations:
        return 0

    n_ops = len(ready_operations)
    
    # Strategy: Critical Path Estimation (CPE)
    # 1. For each ready operation, estimate the "critical path length" of its job.
    #    This is done by summing the minimum possible durations of all remaining operations
    #    in that job. This approximates the earliest possible completion time of the job
    #    if we ignore machine contention (optimistic bound).
    # 2. We prioritize the operation belonging to the job with the longest estimated
    #    remaining critical path. This ensures we keep the "bottleneck" jobs moving.
    # 3. Machine selection: For the chosen operation, select the machine that minimizes
    #    the *estimated completion time* of the job on that machine. 
    #    Actually, a better machine selection heuristic in FJSP is often "Least Loaded" 
    #    or "Fastest". Since we don't have live machine load info in the dispatch dict 
    #    (only static options), we stick to "Fastest" for the selected op, but the 
    #    *selection of the op* is driven by the global job critical path.
    
    # To improve on simple remaining_work, we can look at the *variance* or *bottleneck* 
    # nature. However, remaining_work is already a sum of durations. 
    # Let's refine: 
    # The current program uses normalized remaining_work. 
    # The new strategy: 
    # 1. Calculate estimated completion time for each job = time + remaining_work.
    # 2. Prioritize the job with the max estimated completion time (Critical Job).
    # 3. If multiple ops from the same critical job are ready, pick the one with the 
    #    shortest processing time (SPT) to clear the path quickly? Or the one on the 
    #    fastest machine?
    #    
    # Let's try a different angle: **Congestion-Aware Critical Path**.
    # We don't have machine loads. But we can infer "urgency" differently.
    # 
    # Alternative Structural Change: **Weighted Shortest Processing Time (WSPT)**
    # combined with **Job Criticality**.
    # 
    # Let's implement a "Lookahead" heuristic:
    # For each ready op i in job J_i:
    #   Estimate the completion time of job J_i if we start op i now.
    #   Est_Comp(J_i) = time + duration(op_i, m) + sum(min_duration(op_j) for j in remaining_ops(J_i) excluding op_i)
    #   We want to minimize the makespan, which is roughly max(over all jobs) of Est_Comp.
    #   A common heuristic is to pick the op that minimizes the *maximum* estimated 
    #   completion time of all jobs? No, that's complex.
    #   
    # Simpler and effective: **Critical Path Rule**
    # Pick the operation that belongs to the job with the longest *remaining* critical path.
    # To break ties or add nuance, we can weight by the "slack" or just use the 
    # remaining work.
    #
    # The current program scores: 1.0 * norm_urgency + 0.4 * norm_efficiency.
    # This is a local rule.
    #
    # New Strategy: **Dynamic Critical Path with Machine Selection Optimization**
    # 1. For each ready operation, calculate the "Job Critical Path" (JCP).
    #    JCP is the sum of minimum durations of all remaining operations in that job.
    # 2. Select the operation from the job with the MAXIMUM JCP.
    # 3. For that operation, select the machine that minimizes the *total* estimated 
    #    completion time of the job? 
    #    Actually, let's just select the machine with the minimum duration for the 
    #    selected op, as before, but the *selection of the op* is purely driven by 
    #    the global job critical path, not a normalized blend.
    #
    # Why is this different?
    # The current program normalizes urgency across *all* ready ops. This can lead to 
    # picking a small op from a non-critical job if the critical job's remaining work 
    # is slightly less than the max, but the small op has high efficiency.
    # The new program strictly prioritizes the job that is furthest from completion 
    # (critical path), ensuring the "longest" job is never starved. This is a 
    # "Longest Remaining Processing Time" (LRPT) style heuristic for jobs.
    
    # Pre-compute job critical paths
    job_remaining_work = {}
    for op in ready_operations:
        job_id = op["job"]
        # remaining_work is already provided as the sum of remaining ops? 
        # The doc says "remaining_work". Let's assume it's the sum of durations 
        # of all remaining ops in that job.
        # If multiple ops from the same job are ready, they share the same remaining_work.
        if job_id not in job_remaining_work:
            job_remaining_work[job_id] = op["remaining_work"]
    
    # Find the job with the maximum remaining work (Critical Job)
    max_rem_work = -1
    critical_job = None
    for job_id, work in job_remaining_work.items():
        if work > max_rem_work:
            max_rem_work = work
            critical_job = job_id
            
    # If no critical job found (should not happen), fallback
    if critical_job is None:
        return 0

    # Find all ready operations belonging to the critical job
    candidates = []
    for i, op in enumerate(ready_operations):
        if op["job"] == critical_job:
            candidates.append(i)
            
    if not candidates:
        # Fallback if something went wrong
        return 0
        
    # Among the candidates from the critical job, which one to pick?
    # Heuristic: Pick the one that allows the job to finish fastest?
    # Since they are all in the same job, the remaining work after this op is 
    # (remaining_work - duration(op)).
    # To minimize the completion time of the critical job, we should pick the op 
    # that has the *shortest* duration? No, that doesn't reduce the sum.
    # The sum of remaining work is fixed for the job.
    # However, machine availability might differ.
    # Let's pick the candidate that has the *minimum* processing time on its *best* machine.
    # This clears the bottleneck operation as fast as possible.
    
    best_candidate_idx = -1
    best_min_dur = float('inf')
    best_machine = None
    
    for idx in candidates:
        op = ready_operations[idx]
        min_dur = float('inf')
        best_m = None
        for m, d in op["machine_options"].items():
            if d < min_dur:
                min_dur = d
                best_m = m
        if min_dur < best_min_dur:
            best_min_dur = min_dur
            best_candidate_idx = idx
            best_machine = best_m
            
    if best_candidate_idx == -1:
        return 0

    return (int(best_candidate_idx), best_machine)

# EVOLVE-BLOCK-END
