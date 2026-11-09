"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan.
    Uses Nearest Neighbor Search (greedy construction) with a deadline guard.
    Falls back to NEH (Longest Processing Time) if time is insufficient or for small instances.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin before 120s limit
    
    jobs, machines = processing_times.shape
    
    # Initial heuristic: NEH (Longest Processing Time)
    # This serves as a baseline and fallback.
    total_times = processing_times.sum(axis=1)
    neh_order = list(np.argsort(-total_times))
    
    # If the instance is very small, NEH is likely optimal or close, and NNS overhead isn't worth it.
    # Or if we are already close to deadline.
    if jobs < 5 or time.time() > deadline - 5.0:
        return neh_order

    # Nearest Neighbor Search
    # Current permutation
    current_perm = []
    remaining_jobs = set(range(jobs))
    
    # Precompute some data? 
    # For NNS, at each step, we need to find the job j in remaining_jobs that minimizes makespan(current_perm + [j])
    
    # To speed up, we can maintain the current completion times per machine for the current_perm.
    # C_m = max(C_{m-1}, C_{m-1} + p_{j,m}) ... actually C_m = C_{m-1} + p_{j,m} is wrong.
    # C_m = max(C_{m-1}, C_{m-1}) + p_{j,m} ? No.
    # In flow shop, C_{j,m} = C_{j,m-1} + p_{j,m}.
    # Makespan is C_{last_job, last_machine}.
    # If we append job j to current sequence S, the new makespan is determined by the completion time of j on the last machine.
    # Let C_m be the completion time of the last job in S on machine m.
    # When adding job j:
    # New C'_0 = C_0 + p_{j,0}
    # New C'_m = max(C'_m, C_{m-1}) + p_{j,m} ? No.
    # The completion time of job j on machine m is max(C_{m-1}, C'_{m-1}) + p_{j,m}? 
    # Actually, the start time of job j on machine m is max(C_{m-1} (from previous job on m), C'_{m-1} (from job j on m-1)).
    # So C'_m = max(C_{m-1}, C'_{m-1}) + p_{j,m}.
    
    # We can maintain an array `last_comp` of length `machines` where last_comp[m] is the completion time of the current last job in `current_perm` on machine m.
    # Initially, current_perm is empty, so last_comp is 0.
    
    last_comp = np.zeros(machines)
    
    # We need to evaluate all remaining jobs.
    # To do this efficiently, we can vectorize the calculation of the new makespan for all remaining jobs at once.
    
    # Let R be the matrix of processing times for remaining jobs.
    # For a job j, the new completion times would be:
    # nc_0 = last_comp[0] + R[j, 0]
    # nc_m = max(last_comp[m], nc_{m-1}) + R[j, m]
    # This is a sequential dependency across machines, so it's hard to fully vectorize across machines for all jobs at once in a single NumPy op without looping over machines.
    # However, we can loop over machines (M is usually small) and update a vector of candidate completion times for all remaining jobs.
    
    # Let `cand_comp` be a matrix of shape (num_remaining, machines).
    # At each step, we filter `cand_comp` to only include remaining jobs.
    
    # Optimization: Instead of recalculating from scratch for all remaining jobs every step, we can update the potential completion times incrementally?
    # No, because the "max" operation depends on the previous machine's result which changes for each job.
    # But M is small. Let's just loop over M for each step, updating a vector of length N_remaining.
    
    # Actually, a simpler NNS implementation:
    # For each step k from 1 to N:
    #   For each job j in remaining:
    #     Compute makespan if j is added.
    #   Pick best j.
    
    # This is O(N^2 * M).
    # Let's try to vectorize the inner loop (over jobs) for a fixed step.
    
    # We maintain `last_comp` (shape M).
    # We have a list of `remaining_jobs`.
    # We want to compute the new makespan for each job in `remaining_jobs`.
    
    # Let's create a helper to compute new completion times for a batch of jobs given `last_comp`.
    # batch_p: shape (K, M)
    # last_comp: shape (M,)
    # returns: new_last_comp: shape (K, M) and final_makespan: shape (K,)
    
    def compute_new_completions(batch_p, last_comp):
        # batch_p: (K, M)
        # last_comp: (M,)
        K = batch_p.shape[0]
        # Initialize new completions
        # We need to compute sequentially for each machine.
        # nc[:, 0] = last_comp[0] + batch_p[:, 0]
        # nc[:, m] = max(last_comp[m], nc[:, m-1]) + batch_p[:, m]
        
        nc = np.empty((K, M), dtype=float)
        # Machine 0
        nc[:, 0] = last_comp[0] + batch_p[:, 0]
        for m in range(1, M):
            # nc[:, m] = max(last_comp[m], nc[:, m-1]) + batch_p[:, m]
            nc[:, m] = np.maximum(last_comp[m], nc[:, m-1]) + batch_p[:, m]
        
        # The makespan for each job in the batch is nc[:, -1]
        return nc, nc[:, -1]

    # Main NNS Loop
    current_perm = []
    remaining_indices = list(range(jobs))
    last_comp = np.zeros(machines)
    
    # We will iterate N times.
    # To avoid O(N^2) overhead of list manipulation, we can use a boolean mask or just a list and pop.
    # List pop is O(N). Total O(N^2).
    
    # We need to keep track of the processing times for remaining jobs to pass to compute_new_completions.
    # We can just index processing_times[remaining_indices].
    
    for _ in range(jobs):
        if time.time() > deadline:
            # Timeout! Return best full permutation found so far.
            # Since we are in the middle of NNS, we have a partial permutation.
            # We must return a valid full permutation.
            # Append remaining jobs in their current order (or NEH order) to complete the permutation.
            # This might be suboptimal but valid.
            # Better: fall back to the initial NEH order if we haven't finished NNS?
            # The NEH order is a valid full permutation.
            # If we have spent > 110s, we are in trouble. 
            # But NNS is fast. This should rarely trigger.
            # If it does, returning NEH is safer than a broken partial NNS.
            return neh_order

        if not remaining_indices:
            break
            
        # Get processing times for remaining jobs
        batch_p = processing_times[remaining_indices]
        
        # Compute new completion times and makespans for all remaining jobs
        nc, makespans = compute_new_completions(batch_p, last_comp)
        
        # Find the job with minimum makespan
        min_idx_local = np.argmin(makespans)
        min_job_global = remaining_indices[min_idx_local]
        
        # Update state
        current_perm.append(min_job_global)
        last_comp = nc[min_idx_local]
        
        # Remove chosen job from remaining
        remaining_indices.pop(min_idx_local)
        
        # Check time again?
        # If we are running out of time, we might want to break and return NEH.
        # But we are already in the loop.
        
    return current_perm

# EVOLVE-BLOCK-END
