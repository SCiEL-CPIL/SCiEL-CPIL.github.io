"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _makespan(processing_times, order):
    """
    Compute makespan for a given job order efficiently using NumPy.
    """
    if len(order) == 0:
        return 0.0
    
    n_machines = processing_times.shape[1]
    # Get processing times for the jobs in the given order
    # shape: (n_jobs, n_machines)
    P = processing_times[order]
    
    # Recurrence: C[i, j] = max(C[i-1, j], C[i, j-1]) + P[i, j]
    # We can compute this row by row or using cumulative sums with constraints.
    # A simple and fast way in NumPy for moderate N is to iterate rows but vectorize columns.
    # However, since N can be large, let's try to be as fast as possible.
    
    # c[j] is the completion time of the current job on machine j
    # c_prev[j] is the completion time of the previous job on machine j
    c_prev = np.zeros(n_machines)
    
    for i in range(len(order)):
        p_row = P[i]
        # c_curr[j] = max(c_prev[j], c_curr[j-1]) + p_row[j]
        # This is a sequential dependency along j.
        # We can compute this using a loop over j, which is O(M) per job.
        # Total O(N*M).
        c_curr = np.zeros(n_machines)
        for j in range(n_machines):
            if j == 0:
                c_curr[j] = c_prev[j] + p_row[j]
            else:
                c_curr[j] = max(c_prev[j], c_curr[j-1]) + p_row[j]
        c_prev = c_curr
        
    return c_prev[-1]

def _neh_rule(processing_times):
    """NEH heuristic for flow shop scheduling."""
    n_jobs, n_machines = processing_times.shape
    
    totals = processing_times.sum(axis=1)
    sorted_jobs = list(np.argsort(-totals))
    
    order = []
    for job in sorted_jobs:
        best_pos = 0
        best_ms = float('inf')
        # Try inserting job at every position
        for pos in range(len(order) + 1):
            candidate = order[:pos] + [job] + order[pos:]
            ms = _makespan(processing_times, candidate)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        order.insert(best_pos, job)
    
    return order

def _local_search_insertion(processing_times, order, time_limit, start_time):
    """
    Local search with insertion moves.
    Remove a job and re-insert it at the best position.
    """
    n_jobs = len(order)
    if n_jobs <= 1:
        return order, _makespan(processing_times, order)
    
    best_order = list(order)
    best_ms = _makespan(processing_times, best_order)
    
    while True:
        if time.time() - start_time > time_limit:
            break
        
        improved = False
        
        # To avoid getting stuck in a specific local optimum or order bias,
        # we can randomize the order of jobs to consider.
        # But standard first-improvement often works well with a fixed or random scan.
        # Let's use a random scan for diversity.
        jobs_to_consider = list(range(n_jobs))
        random.shuffle(jobs_to_consider)
        
        for i in jobs_to_consider:
            if time.time() - start_time > time_limit:
                break
            
            job = best_order[i]
            # Remove job
            remaining = best_order[:i] + best_order[i+1:]
            
            # Try inserting at each position
            best_ins_pos = -1
            best_ins_ms = best_ms
            
            # We can skip positions that are equivalent to the original position
            # Original position in 'remaining' is i (if i < len(remaining)) or len(remaining) (if i == n_jobs-1)
            # Actually, removing item at i shifts indices.
            # If we insert back at i, it's the same.
            # We only need to check positions 0 to len(remaining)-1, excluding the "hole" position if it's valid?
            # Simpler: check all positions 0 to len(remaining).
            # If pos == i, the order is unchanged.
            
            for pos in range(len(remaining) + 1):
                if pos == i:
                    continue
                
                candidate = remaining[:pos] + [job] + remaining[pos:]
                ms = _makespan(processing_times, candidate)
                
                if ms < best_ins_ms:
                    best_ins_ms = ms
                    best_ins_pos = pos
            
            if best_ins_pos != -1:
                # Apply the best move found for this job
                # Reconstruct the order
                remaining = best_order[:i] + best_order[i+1:]
                best_order = remaining[:best_ins_pos] + [job] + remaining[best_ins_pos:]
                best_ms = best_ins_ms
                improved = True
                # Continue scanning other jobs, or restart?
                # First improvement usually continues to the next job.
                # Since the order changed, the indices for subsequent 'i' in jobs_to_consider
                # might be slightly off if we don't adjust, but since we just pick a job index 'i'
                # from the original range, and 'i' refers to the position in the *current* best_order,
                # we should be careful.
                # Wait, 'i' is an index into best_order. If we modify best_order, the meaning of 'i' changes.
                # It is safer to restart the scan or just accept that we are doing a "best improvement" per job
                # but the indices in the loop are static.
                # Let's restart the scan to be safe and effective.
                break
        
        if not improved:
            break
            
    return best_order, best_ms

def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan on flow-shop instances.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs <= 1:
        return list(range(n_jobs))
    
    # 1. Generate initial solutions
    # NEH is a strong heuristic
    order_neh = _neh_rule(processing_times)
    ms_neh = _makespan(processing_times, order_neh)
    
    # Add a few random starts to escape local optima
    best_order = order_neh
    best_ms = ms_neh
    
    # Try a few random restarts if time permits
    # Each random start + local search is expensive, so we limit it.
    # Let's do 1 random start if n_jobs is small enough or if NEH is slow?
    # NEH is O(N^2 * N * M) = O(N^3 M).
    # For N=100, M=10, that's 10^6 * 10 = 10^7 ops. Fast.
    
    # Let's just do one random start to diversify.
    order_rand = list(range(n_jobs))
    random.shuffle(order_rand)
    ms_rand = _makespan(processing_times, order_rand)
    
    if ms_rand < best_ms:
        best_order = order_rand
        best_ms = ms_rand

    # 2. Local Search Refinement
    # Use Insertion Local Search
    remaining_time = deadline - time.time()
    if remaining_time > 5.0:
        time_limit_ls = remaining_time - 5.0
        best_order, best_ms = _local_search_insertion(processing_times, best_order, time_limit_ls, start_time)
        
        # If we still have time, do another round?
        # Insertion local search can get stuck. A second round might not help if it's at a local optimum.
        # But maybe a random perturbation + local search (memetic) would help.
        
        remaining_time = deadline - time.time()
        if remaining_time > 10.0:
            # Perturb the best solution
            perturbed_order = list(best_order)
            # Swap two random jobs
            i, j = random.sample(range(n_jobs), 2)
            perturbed_order[i], perturbed_order[j] = perturbed_order[j], perturbed_order[i]
            
            # Run local search on perturbed solution
            time_limit_ls2 = remaining_time - 5.0
            perturbed_order, ms_perturbed = _local_search_insertion(processing_times, perturbed_order, time_limit_ls2, start_time)
            
            if ms_perturbed < best_ms:
                best_order = perturbed_order
                best_ms = ms_perturbed

    return best_order
# EVOLVE-BLOCK-END
