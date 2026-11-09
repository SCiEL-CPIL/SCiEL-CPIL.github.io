"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _makespan(processing_times, order):
    """Compute makespan for a given job order."""
    if len(order) == 0:
        return 0.0
    n_jobs = len(order)
    n_machines = processing_times.shape[1]
    
    # Efficient O(N*M) calculation
    # prev_completions[m] stores the completion time of the previous job on machine m
    prev_completions = np.zeros(n_machines)
    makespan = 0.0
    
    for job_idx in order:
        curr_job_completion = 0.0
        for m in range(n_machines):
            # C(i, m) = max(C(i-1, m), C(i, m-1)) + P(i, m)
            # C(i-1, m) is prev_completions[m]
            # C(i, m-1) is curr_job_completion (from previous machine in this job)
            
            val = max(prev_completions[m], curr_job_completion) + processing_times[job_idx, m]
            curr_job_completion = val
            prev_completions[m] = val
        
        makespan = curr_job_completion
        
    return makespan


def _cds_rule(processing_times):
    """
    Generate initial solutions using heuristics similar to CDS.
    Returns a list of candidate orders.
    """
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0: 
        return [[]]
    if n_jobs == 1: 
        return [[0]]
    
    total_pt = processing_times.sum(axis=1)
    
    # 1. NEH Heuristic
    sorted_jobs = np.argsort(-total_pt)
    order_neh = [int(sorted_jobs[0])]
    for k in range(1, n_jobs):
        job = int(sorted_jobs[k])
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(order_neh) + 1):
            cand = order_neh[:pos] + [job] + order_neh[pos:]
            ms = _makespan(processing_times, cand)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        order_neh = order_neh[:best_pos] + [job] + order_neh[best_pos:]
    
    # 2. LPT (Longest Processing Time first)
    order_lpt = list(np.argsort(-total_pt).astype(int))
    
    # 3. SPT (Shortest Processing Time first)
    order_spt = list(np.argsort(total_pt).astype(int))
    
    return [order_neh, order_lpt, order_spt]


def _simulated_annealing(processing_times, initial_order, time_limit=10.0):
    """
    Improve the initial order using Simulated Annealing.
    """
    n_jobs, n_machines = processing_times.shape
    if n_jobs <= 1:
        return initial_order
    
    best_order = list(initial_order)
    best_ms = _makespan(processing_times, best_order)
    current_order = list(best_order)
    current_ms = best_ms
    
    # Initialize Temperature
    rng = np.random.RandomState(42)
    random_order = list(rng.permutation(n_jobs))
    random_ms = _makespan(processing_times, random_order)
    
    if random_ms > best_ms:
        T0 = (random_ms - best_ms) * 1.5
    else:
        T0 = best_ms * 0.5 + 1.0
    
    T = T0
    alpha = 0.99999 # Cooling rate
    
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        # Generate neighbor: remove a job, insert at random position
        i = rng.randint(n_jobs)
        job_to_move = current_order.pop(i)
        
        # Insert at random position
        j = rng.randint(n_jobs)
        current_order.insert(j, job_to_move)
        
        new_ms = _makespan(processing_times, current_order)
        delta = new_ms - current_ms
        
        if delta < 0:
            # Improvement
            current_ms = new_ms
            if new_ms < best_ms:
                best_ms = new_ms
                best_order = list(current_order)
        else:
            # Worse or equal
            if T > 0:
                prob = np.exp(-delta / T)
                if rng.rand() < prob:
                    current_ms = new_ms
            
        # Cool down
        T *= alpha
        
    return best_order


def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan using CDS initialization + Simulated Annealing.
    """
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # Generate initial candidates
    candidates = _cds_rule(processing_times)
    
    # Add a random candidate
    rng = np.random.RandomState(42)
    candidates.append(list(rng.permutation(n_jobs)))
    
    # Find the best initial candidate
    initial_best = None
    initial_best_ms = float('inf')
    for cand in candidates:
        ms = _makespan(processing_times, cand)
        if ms < initial_best_ms:
            initial_best_ms = ms
            initial_best = cand
    
    # Run SA on the best initial candidate
    # 10 seconds is a safe budget within the 120s limit
    sa_order = _simulated_annealing(processing_times, initial_best, time_limit=10.0)
    
    return sa_order
# EVOLVE-BLOCK-END
