"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _makespan(processing_times, order):
    """Compute makespan for a given job order efficiently."""
    n_jobs = len(order)
    if n_jobs == 0:
        return 0.0
    n_machines = processing_times.shape[1]
    
    # Use vectorized computation where possible
    # C[j][m] = completion time of job j on machine m
    # We can compute this incrementally
    completion = np.zeros((n_jobs, n_machines), dtype=np.float64)
    
    for j in range(n_jobs):
        job_idx = order[j]
        for m in range(n_machines):
            p = processing_times[job_idx, m]
            if m == 0:
                if j == 0:
                    completion[j, m] = p
                else:
                    completion[j, m] = completion[j-1, m] + p
            else:
                if j == 0:
                    completion[j, m] = completion[j, m-1] + p
                else:
                    completion[j, m] = max(completion[j-1, m], completion[j, m-1]) + p
    
    return completion[-1, -1]


def _neh_rule(processing_times):
    """NEH heuristic for flow-shop scheduling."""
    n_jobs, n_machines = processing_times.shape
    
    # Sort jobs by total processing time in descending order
    total_pt = processing_times.sum(axis=1)
    sorted_jobs = np.argsort(-total_pt)
    
    # Build the permutation incrementally
    permutation = [sorted_jobs[0]]
    
    for job in sorted_jobs[1:]:
        # Try inserting the job at each position
        best_pos = 0
        best_ms = float('inf')
        
        for pos in range(len(permutation) + 1):
            new_perm = permutation[:pos] + [job] + permutation[pos:]
            ms = _makespan(processing_times, new_perm)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        
        permutation = permutation[:best_pos] + [job] + permutation[best_pos:]
    
    return permutation


def _local_search(processing_times, order, max_iter=100):
    """2-opt style local search with adjacent swaps."""
    n_jobs = len(order)
    current_order = list(order)
    current_ms = _makespan(processing_times, current_order)
    
    improved = True
    iter_count = 0
    while improved and iter_count < max_iter:
        improved = False
        iter_count += 1
        
        for i in range(n_jobs - 1):
            # Try swapping adjacent jobs i and i+1
            new_order = current_order[:]
            new_order[i], new_order[i+1] = new_order[i+1], new_order[i]
            
            new_ms = _makespan(processing_times, new_order)
            
            if new_ms < current_ms:
                current_order = new_order
                current_ms = new_ms
                improved = True
    
    return current_order, current_ms


def _simulated_annealing(processing_times, initial_order, start_temp=10.0, end_temp=0.01, 
                          cooling_rate=0.95, max_iter=500, deadline=None):
    """Simulated annealing with adjacent swaps."""
    n_jobs = len(initial_order)
    current_order = list(initial_order)
    current_ms = _makespan(processing_times, current_order)
    best_order = list(current_order)
    best_ms = current_ms
    
    temp = start_temp
    iter_count = 0
    
    while temp > end_temp and iter_count < max_iter:
        iter_count += 1
        
        # Check deadline
        if deadline is not None and time.time() > deadline:
            break
        
        # Generate a random adjacent swap
        i = np.random.randint(0, n_jobs - 1)
        j = i + 1
        
        new_order = current_order[:]
        new_order[i], new_order[j] = new_order[j], new_order[i]
        
        new_ms = _makespan(processing_times, new_order)
        delta = new_ms - current_ms
        
        # Accept if better or with probability exp(-delta/temp)
        if delta < 0 or np.random.random() < np.exp(-delta / max(temp, 1e-10)):
            current_order = new_order
            current_ms = new_ms
            
            if current_ms < best_ms:
                best_ms = current_ms
                best_order = list(current_order)
        
        temp *= cooling_rate
    
    return best_order, best_ms


def order_jobs(processing_times):
    """Construct a job permutation minimizing makespan on flow-shop instances."""
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n_jobs, n_machines = processing_times.shape
    
    # Handle edge cases
    if n_jobs <= 1:
        return list(range(n_jobs))
    if n_machines == 0:
        return list(range(n_jobs))
    
    # Start with NEH heuristic
    neh_order = _neh_rule(processing_times)
    best_order = list(neh_order)
    best_ms = _makespan(processing_times, best_order)
    
    # Check if we should continue
    if time.time() > deadline:
        return best_order
    
    # Local search refinement
    ls_order, ls_ms = _local_search(processing_times, best_order, max_iter=50)
    if ls_ms < best_ms:
        best_order = ls_order
        best_ms = ls_ms
    
    if time.time() > deadline:
        return best_order
    
    # Simulated annealing for further improvement
    sa_order, sa_ms = _simulated_annealing(processing_times, best_order, 
                                            start_temp=5.0, end_temp=0.05,
                                            cooling_rate=0.97, max_iter=300,
                                            deadline=deadline)
    if sa_ms < best_ms:
        best_order = sa_order
        best_ms = sa_ms
    
    # Final local search on the SA result
    if time.time() < deadline - 5:
        final_order, final_ms = _local_search(processing_times, best_order, max_iter=30)
        if final_ms < best_ms:
            best_order = final_order
            best_ms = final_ms
    
    return best_order
# EVOLVE-BLOCK-END
