"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def _compute_makespan(processing_times, order):
    """
    Efficiently compute the makespan for a given job order.
    processing_times: (n_jobs, n_machines)
    order: list of job indices
    """
    n_machines = processing_times.shape[1]
    if len(order) == 0:
        return 0.0
    
    # completion[j] stores the completion time of the last processed job on machine j
    completion = np.zeros(n_machines)
    
    for job_idx in order:
        current_p = processing_times[job_idx]
        for j in range(n_machines):
            if j == 0:
                completion[j] += current_p[j]
            else:
                completion[j] = max(completion[j-1], completion[j]) + current_p[j]
                
    return completion[-1]

def _simulated_annealing(processing_times, initial_order, max_iterations=2000, initial_temp=100.0, cooling_rate=0.995, rng=None):
    """
    Perform Simulated Annealing to minimize makespan.
    """
    if rng is None:
        rng = np.random.RandomState(42)
        
    n_jobs = len(initial_order)
    if n_jobs <= 1:
        return initial_order
    
    current_order = list(initial_order)
    current_makespan = _compute_makespan(processing_times, current_order)
    best_order = list(current_order)
    best_makespan = current_makespan
    
    temp = initial_temp
    
    for i in range(max_iterations):
        # Generate a neighbor
        if rng.rand() < 0.5:
            # Swap
            i1 = rng.randint(n_jobs)
            i2 = rng.randint(n_jobs)
            if i1 == i2:
                continue
            neighbor_order = current_order.copy()
            neighbor_order[i1], neighbor_order[i2] = neighbor_order[i2], neighbor_order[i1]
        else:
            # Insert
            i1 = rng.randint(n_jobs)
            i2 = rng.randint(n_jobs)
            if i1 == i2:
                continue
            neighbor_order = current_order.copy()
            job = neighbor_order.pop(i1)
            neighbor_order.insert(i2, job)
            
        neighbor_makespan = _compute_makespan(processing_times, neighbor_order)
        
        # Acceptance probability
        delta = neighbor_makespan - current_makespan
        if delta < 0:
            current_order = neighbor_order
            current_makespan = neighbor_makespan
            if current_makespan < best_makespan:
                best_order = list(current_order)
                best_makespan = current_makespan
        else:
            if temp > 1e-10:
                prob = np.exp(-delta / temp)
                if rng.rand() < prob:
                    current_order = neighbor_order
                    current_makespan = neighbor_makespan
        
        # Cool down
        temp *= cooling_rate
        
    return best_order

def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan using Simulated Annealing.
    """
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    rng = np.random.RandomState(42)
    
    # Generate a good initial solution using LPT
    total_pt = processing_times.sum(axis=1)
    initial_order = list(np.argsort(-total_pt))
    
    # Run SA
    max_iters = max(1000, n_jobs * 20)
    init_temp = np.max(processing_times) * 10
    
    best_order = _simulated_annealing(processing_times, initial_order, max_iterations=max_iters, initial_temp=init_temp, cooling_rate=0.995, rng=rng)
    
    return best_order
# EVOLVE-BLOCK-END
