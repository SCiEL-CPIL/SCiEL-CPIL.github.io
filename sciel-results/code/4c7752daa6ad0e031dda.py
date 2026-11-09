"""
Flow-shop permutation construction using Simulated Annealing on a continuous priority vector.
Strategy:
1. Initialize with NEH heuristic to get a strong starting permutation.
2. Convert the permutation to a continuous priority vector (weights).
3. Use Simulated Annealing to perturb the weights, re-sort to get a new permutation, and evaluate.
4. The continuous space allows for larger jumps in the discrete permutation space, helping escape local optima.
5. Use a fast makespan calculation.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _makespan_vec(order, P):
    """
    Compute makespan for a given job order using vectorized NumPy operations.
    P: (n_jobs, n_machines)
    order: list/array of job indices
    """
    if len(order) == 0:
        return 0.0
    
    n_jobs_seq = len(order)
    n_machines = P.shape[1]
    
    # Extract processing times for the sequence
    seq_P = P[order]
    
    # C_machines[j] will hold the completion time of job j on the current machine.
    c_machines = np.zeros(n_machines)
    last_completion = 0.0
    
    for j in range(n_jobs_seq):
        row = seq_P[j]
        c_curr = 0.0
        for m in range(n_machines):
            c_curr = max(c_machines[m], c_curr) + row[m]
            c_machines[m] = c_curr
        last_completion = c_curr
        
    return last_completion

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

def order_jobs(processing_times):
    """
    Order jobs to minimize makespan using NEH initialization and Simulated Annealing on priority vectors.
    """
    P = np.asarray(processing_times, dtype=np.float64)
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
    
    # 1. Initialize with NEH
    current_seq = _neh_heuristic(P)
    current_makespan = _makespan_vec(current_seq, P)
    
    best_seq = current_seq.copy()
    best_makespan = current_makespan
    
    # Initialize priority vector
    # We want the current sequence to correspond to this vector.
    # If we sort indices by priority, we get the sequence.
    # So priority[job] should be such that job at pos 0 has highest priority?
    # Usually, we sort in descending order of priority to get the sequence.
    # Let's define: seq = argsort(-priority).
    # So priority[seq[0]] > priority[seq[1]] > ...
    # We can assign priorities based on rank.
    priority = np.zeros(n_jobs)
    for rank, job in enumerate(current_seq):
        priority[job] = n_jobs - rank  # Higher rank (earlier) -> Higher priority
    
    # Simulated Annealing Parameters
    T0 = 100.0  # Initial temperature
    Tend = 1.0  # Final temperature
    alpha = 0.995 # Cooling rate
    
    # Determine number of iterations based on time, but we'll use a while loop with time check.
    # Perturbation scale
    sigma = 5.0
    
    T = T0
    iter_count = 0
    
    # Pre-compute something? No, just run.
    
    while (time.time() - start_time) < time_limit:
        iter_count += 1
        
        # Perturb priority vector
        # Add Gaussian noise
        # Scale noise by temperature? Or keep constant and rely on T for acceptance?
        # Standard SA: x_new = x_old + N(0, sigma). Accept if better or with prob exp(-delta/T).
        # Here x is the vector.
        
        new_priority = priority + np.random.normal(0, sigma, n_jobs)
        
        # Get new sequence
        # Sort by priority descending
        new_seq = np.argsort(-new_priority).tolist()
        
        # Check if sequence changed. If not, skip?
        # Actually, if priorities are very close, order might not change.
        # We can check if new_seq == current_seq.
        # But argsort is stable? No, we should ensure distinct priorities or handle ties.
        # With continuous noise, ties are unlikely.
        
        if new_seq == current_seq:
            continue
            
        new_makespan = _makespan_vec(new_seq, P)
        
        delta = new_makespan - current_makespan
        
        if delta < 0:
            # Accept
            current_seq = new_seq
            current_makespan = new_makespan
            priority = new_priority
            
            if current_makespan < best_makespan:
                best_makespan = current_makespan
                best_seq = current_seq.copy()
        else:
            # Accept with probability
            if T > 0:
                prob = np.exp(-delta / T)
                if np.random.random() < prob:
                    current_seq = new_seq
                    current_makespan = new_makespan
                    priority = new_priority
        
        # Cool down
        T *= alpha
        
        # If T is too low, restart? Or just stop?
        # Let's keep cooling.
        
        # Dynamic sigma?
        # If we are stuck, maybe increase sigma?
        # For simplicity, keep static.
        
    return best_seq
# EVOLVE-BLOCK-END
