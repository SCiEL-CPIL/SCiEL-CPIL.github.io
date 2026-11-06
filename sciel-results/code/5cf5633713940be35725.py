"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan using Simulated Annealing.
    
    1. Generate an initial solution using the NEH heuristic.
    2. Refine the solution using Simulated Annealing with insertion moves.
    """
    n_jobs, n_machines = processing_times.shape
    
    # --- Stage 1: NEH Heuristic for Initial Solution ---
    seq = _neh_heuristic(processing_times)
    current_cost = _calc_makespan(processing_times, seq)
    best_seq = seq.copy()
    best_cost = current_cost
    
    # --- Stage 2: Simulated Annealing ---
    # Time budget: 110 seconds to stay within 120s limit
    start_time = time.time()
    time_limit = 110.0
    
    # SA Parameters
    initial_temp = 1000.0
    final_temp = 1.0
    current_temp = initial_temp
    cooling_rate = 0.99999  # Tuned for ~100k-1M iterations
    
    while time.time() - start_time < time_limit:
        # Generate a neighbor by inserting a random job into a random position
        i = random.randint(0, n_jobs - 1)
        j = random.randint(0, n_jobs - 1)
        
        if i == j:
            continue
            
        job = seq[i]
        
        # Create candidate sequence by removing job at i and inserting at j
        # Remove from i
        cand = seq[:i] + seq[i+1:]
        
        # Adjust insertion index j if j > i
        if j > i:
            j -= 1
            
        # Insert at j
        cand = cand[:j] + [job] + cand[j:]
        
        new_cost = _calc_makespan(processing_times, cand)
        delta_cost = new_cost - current_cost
        
        # Accept if better or with probability exp(-delta/T)
        if delta_cost < 0 or (current_temp > 0 and random.random() < np.exp(-delta_cost / current_temp)):
            seq = cand
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_seq = seq.copy()
        
        # Cool down
        current_temp *= cooling_rate
        if current_temp < final_temp:
            current_temp = final_temp

    return best_seq


def _neh_heuristic(processing_times):
    """NEH heuristic for initial solution."""
    n_jobs, n_machines = processing_times.shape
    total_times = processing_times.sum(axis=1)
    order = list(np.argsort(-total_times))
    
    seq = [order[0]]
    for j in order[1:]:
        best_pos = 0
        best_makespan = float('inf')
        
        for pos in range(len(seq) + 1):
            candidate = seq[:pos] + [j] + seq[pos:]
            makespan = _calc_makespan(processing_times, candidate)
            if makespan < best_makespan:
                best_makespan = makespan
                best_pos = pos
        
        seq = seq[:best_pos] + [j] + seq[best_pos:]
    
    return seq


def _calc_makespan(processing_times, seq):
    """Calculate the makespan for a given job sequence."""
    n_machines = processing_times.shape[1]
    completion_times = np.zeros(n_machines)
    
    for job_idx in seq:
        for m in range(n_machines):
            if m == 0:
                completion_times[m] += processing_times[job_idx, m]
            else:
                completion_times[m] = max(completion_times[m], completion_times[m-1]) + processing_times[job_idx, m]
    
    return completion_times[-1]
# EVOLVE-BLOCK-END
