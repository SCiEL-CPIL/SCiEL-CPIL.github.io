"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

# Deadline guard
_DEADLINE_START = _time.time()
_DEADLINE_LIMIT = 145.0  # safety margin before 150s kill


def _check_deadline():
    return (_time.time() - _DEADLINE_START) > _DEADLINE_LIMIT


def dispatch(ready_operations, time):
    if not ready_operations:
        return 0
    
    n = len(ready_operations)
    
    # If only one option, return it
    if n == 1:
        return 0
    
    # Deadline check - if we're close to timeout, fall back to simple rule
    if _check_deadline():
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))

    # 1. Calculate the current global makespan lower bound:
    #    The maximum remaining work across all jobs.
    #    This represents the theoretical minimum time left if no machine conflicts existed.
    current_max_remaining = max(op["remaining_work"] for op in ready_operations)
    
    # 2. For each ready operation, simulate its completion to estimate the new lower bound.
    #    Completing an operation reduces that job's remaining work by the duration of the operation.
    #    We want to pick the operation that most significantly reduces this global bottleneck.
    
    best_score = -np.inf
    best_index = 0
    
    for i, op in enumerate(ready_operations):
        # Use the minimum possible duration for this operation to estimate the maximum possible reduction
        # This is a conservative "best case" for reducing the critical path quickly.
        min_dur = min(op["machine_options"].values()) if op["machine_options"] else 0.0
        
        # If this operation is on the critical path (i.e., its job has the max remaining work),
        # completing it will directly reduce the global lower bound.
        # If it's not on the critical path, the global lower bound (max of others) might stay the same
        # or drop if this job was tied for max.
        
        # To be precise without full simulation:
        # New remaining work for this job = op["remaining_work"] - min_dur
        # The new global max will be max(
        #     max(remaining_work of all OTHER jobs), 
        #     op["remaining_work"] - min_dur
        # )
        
        # Optimization: We can approximate the "max of others" by looking at the distribution.
        # However, a simpler and robust proxy for "Critical Path Reduction" is:
        # How much does this operation reduce the *maximum* remaining work?
        
        # Let's compute the reduction in the specific job's work, weighted by how close it is to the max.
        # Actually, the most direct look-ahead is:
        # If I do this op, what is the new max remaining work?
        
        # To avoid O(N^2) per dispatch (though N is small, ~10-50), we can pre-compute the max of others.
        # But since N is small in typical job-shop dispatch steps (ready set is usually small), 
        # a simple loop or vectorized approach is fine.
        
        # Let's use a vectorized approach for the whole set to be fast.
        pass
    
    # Vectorized approach for speed:
    remaining_works = np.array([op["remaining_work"] for op in ready_operations], dtype=float)
    min_durations = np.array([
        min(op["machine_options"].values()) if op["machine_options"] else 0.0
        for op in ready_operations
    ], dtype=float)
    
    # For each operation i, if we complete it:
    # new_work_i = remaining_works[i] - min_durations[i]
    # new_global_max_i = max(
    #     max(remaining_works[j] for j != i), 
    #     new_work_i
    # )
    
    # We want to MINIMIZE new_global_max_i.
    # Equivalently, we want to MAXIMIZE (current_global_max - new_global_max_i).
    
    # Let's calculate the "max of others" for each i.
    # This is the second largest element in remaining_works, unless the largest is unique and belongs to i.
    
    # Find the max and second max of remaining_works
    # If max is unique, max_of_others for the max element is second_max, for others it is max.
    # If max is not unique, max_of_others is max for all.
    
    max_work = np.max(remaining_works)
    max_indices = np.where(remaining_works == max_work)[0]
    
    if len(max_indices) > 1:
        # Multiple jobs have the max remaining work.
        # For any of them, completing it leaves at least one other job with max_work.
        # So max_of_others is max_work for all.
        max_of_others = np.full(n, max_work)
    else:
        # Unique max.
        # The job with max_work has max_of_others = second_max.
        # All other jobs have max_of_others = max_work.
        # Find second max
        remaining_works_copy = remaining_works.copy()
        remaining_works_copy[max_indices[0]] = -np.inf
        second_max = np.max(remaining_works_copy) if n > 1 else 0.0
        
        max_of_others = np.full(n, max_work)
        max_of_others[max_indices[0]] = second_max
        
    # Calculate new remaining work for each job if its op is completed
    new_remaining_works = remaining_works - min_durations
    
    # Calculate new global max for each scenario
    new_global_maxs = np.maximum(max_of_others, new_remaining_works)
    
    # Score: higher reduction in makespan lower bound is better.
    # Reduction = current_max_work - new_global_maxs
    reductions = current_max_work - new_global_maxs
    
    # Tie-breaker: If reductions are equal, prefer shorter duration (frees machine sooner)
    # and then by job ID for determinism.
    # We want to maximize reduction.
    # If reduction is 0 (e.g., not on critical path), we fall back to other heuristics.
    
    # Add a small bonus for shorter duration to break ties among non-critical path ops
    # Normalize duration to [0,1] for the bonus
    min_dur_min = np.min(min_durations)
    min_dur_max = np.max(min_durations)
    if min_dur_max - min_dur_min > 1e-9:
        norm_dur = (min_durations - min_dur_min) / (min_dur_max - min_dur_min)
    else:
        norm_dur = np.zeros(n)
        
    # Bonus for shorter duration (invert norm_dur)
    duration_bonus = 0.1 * (1.0 - norm_dur)
    
    # Add small job ID tie-breaker
    job_ids = np.array([op["job"] for op in ready_operations], dtype=float)
    job_id_bonus = 1e-6 * (1.0 - (job_ids - np.min(job_ids)) / (np.max(job_ids) - np.min(job_ids) + 1e-9))
    
    final_scores = reductions + duration_bonus + job_id_bonus
    
    best_index = int(np.argmax(final_scores))
    
    return best_index
# EVOLVE-BLOCK-END
