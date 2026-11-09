"""
Flow-shop permutation construction using Tabu Search with a priority-list representation.
Strategy:
1. Initialize with the NEH heuristic for a strong starting point.
2. Use a Tabu Search that operates on a continuous vector of job weights.
3. The priority-list representation maps weights to a permutation via sorting.
4. Tabu list prevents cycling by forbidding recent weight vector states.
5. Aspiration criteria allow accepting tabu moves if they beat the best solution.
6. This differs from NEH+local search by using memory-based diversification via continuous weights.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _makespan_vec(order, P):
    """
    Compute makespan for a given job order using a loop over jobs and machines.
    P: (n_jobs, n_machines)
    order: list/array of job indices
    """
    if len(order) == 0:
        return 0.0
    
    n_jobs_seq = len(order)
    n_machines = P.shape[1]
    
    seq_P = P[order]
    c_machines = np.zeros(n_machines)
    
    for j in range(n_jobs_seq):
        row = seq_P[j]
        c_curr = 0.0
        for m in range(n_machines):
            c_curr = max(c_machines[m], c_curr) + row[m]
            c_machines[m] = c_curr
    
    return c_curr

def _neh_heuristic(P):
    """
    NEH heuristic for initial solution.
    """
    n_jobs, n_machines = P.shape
    totals = P.sum(axis=1)
    order_indices = np.argsort(-totals)
    
    current_seq = []
    for idx in order_indices:
        best_pos = 0
        best_makespan = float('inf')
        
        for pos in range(len(current_seq) + 1):
            new_seq = current_seq[:pos] + [idx] + current_seq[pos:]
            ms = _makespan_vec(new_seq, P)
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
        
        current_seq.insert(best_pos, idx)
        
    return current_seq

def _weights_to_permutation(weights):
    """
    Convert a vector of job weights to a permutation.
    Jobs are sorted in descending order of weights.
    """
    n = len(weights)
    # Use argsort with a stable tie-breaking to ensure reproducibility
    # Sort indices by weight descending; if weights are equal, keep original order
    indices = np.arange(n)
    # We want descending order of weights
    order = np.argsort(-weights, kind='stable')
    return list(order)

def _tabu_search(P, time_limit):
    """
    Tabu Search using a priority-list (continuous weight vector) representation.
    """
    n_jobs, n_machines = P.shape
    start_time = time.time()
    
    # Initialize weights based on NEH order
    neh_seq = _neh_heuristic(P)
    
    # Create initial weights: rank-based, with NEH order getting higher weights
    # Job at position 0 in NEH gets highest weight, etc.
    init_weights = np.zeros(n_jobs)
    for pos, job in enumerate(neh_seq):
        init_weights[job] = n_jobs - pos
    
    current_weights = init_weights.copy()
    current_seq = _weights_to_permutation(current_weights)
    current_makespan = _makespan_vec(current_seq, P)
    
    best_seq = current_seq.copy()
    best_makespan = current_makespan
    
    # Tabu list: store recent weight vectors (or a hash of them)
    tabu_list = []
    tabu_tenure = max(5, n_jobs // 2)  # How many iterations a move stays tabu
    
    # Parameters
    step_size = 1.0  # Perturbation size for weight vector
    max_iter = 50000
    
    i = 0
    while i < max_iter:
        # Check time
        if (time.time() - start_time) > time_limit:
            break
        
        i += 1
        
        # Generate a neighbor by perturbing the weight vector
        new_weights = current_weights.copy()
        
        # Perturb a random subset of weights
        n_perturb = max(1, min(n_jobs, random.randint(1, max(1, n_jobs // 3))))
        jobs_to_perturb = random.sample(range(n_jobs), n_perturb)
        for job in jobs_to_perturb:
            new_weights[job] += random.uniform(-step_size, step_size) * (n_jobs + 1)
        
        # Convert to permutation
        new_seq = _weights_to_permutation(new_weights)
        new_makespan = _makespan_vec(new_seq, P)
        
        # Check tabu status
        # Use a simple hash of the weight vector (rounded to avoid floating point issues)
        weight_hash = hash(tuple(np.round(new_weights, 4).tolist()))
        
        is_tabu = weight_hash in tabu_list
        
        # Aspiration criterion: accept if it beats the best
        aspiration = new_makespan < best_makespan
        
        # Accept if not tabu, or if aspiration is met
        if not is_tabu or aspiration:
            current_weights = new_weights
            current_seq = new_seq
            current_makespan = new_makespan
            
            if current_makespan < best_makespan:
                best_makespan = current_makespan
                best_seq = current_seq.copy()
        
        # Update tabu list
        tabu_list.append(weight_hash)
        if len(tabu_list) > tabu_tenure:
            tabu_list.pop(0)
        
        # Adaptive step size: decrease over time
        step_size *= 0.999
    
    return best_seq

def order_jobs(processing_times):
    """
    Order jobs to minimize makespan using Tabu Search with priority-list representation.
    """
    P = np.asarray(processing_times)
    if P.size == 0:
        return []
        
    n_jobs, n_machines = P.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
        
    start_time = time.time()
    # 150s limit, use 145s to be safe
    time_limit = 145.0
    
    # Run Tabu Search
    best_seq = _tabu_search(P, time_limit)
    
    return best_seq
# EVOLVE-BLOCK-END
