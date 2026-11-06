"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linprog


def _makespan(order, processing_times):
    """Calculate the makespan for a given job order."""
    n_jobs = len(order)
    n_machines = processing_times.shape[1]
    # completion_times[m] is the completion time of the current job on machine m
    # We simulate the flow shop.
    # Let C[j][m] be the completion time of job j on machine m.
    # C[j][0] = C[j-1][0] + P[j][0]
    # C[j][m] = max(C[j][m-1], C[j-1][m]) + P[j][m]
    
    # Using a 1D array for the previous job's completion times on each machine
    prev_completion = np.zeros(n_machines)
    current_completion = np.zeros(n_machines)
    
    for j in order:
        # Machine 0
        current_completion[0] = prev_completion[0] + processing_times[j, 0]
        
        # Other machines
        for m in range(1, n_machines):
            current_completion[m] = max(current_completion[m-1], prev_completion[m]) + processing_times[j, m]
        
        prev_completion = current_completion.copy()
        
    return prev_completion[-1]


def order_jobs(processing_times):
    n_jobs, n_machines = processing_times.shape
    
    # If small, we could try exact, but let's stick to heuristics + local search
    # which is robust and fast.
    
    # 1. Generate initial solution using NEH heuristic (Nawaz-Enscore-Ham)
    # NEH is generally better than simple sum sorting.
    
    # Step 1: Compute total processing time for each job
    total_times = processing_times.sum(axis=1)
    
    # Step 2: Sort jobs in decreasing order of total processing time
    sorted_jobs = np.argsort(-total_times)
    
    # Step 3: Insert each job into the best position in the sequence
    sequence = []
    
    for job in sorted_jobs:
        best_pos = 0
        best_makespan = float('inf')
        
        # Try inserting job at every possible position in the current sequence
        for pos in range(len(sequence) + 1):
            candidate_seq = sequence[:pos] + [job] + sequence[pos:]
            
            # Calculate makespan for this candidate sequence
            # Optimization: We only need to check if it's better.
            # For small N, full calc is fine.
            ms = _makespan(candidate_seq, processing_times)
            
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
        
        sequence.insert(best_pos, job)
    
    best_sequence = sequence
    best_makespan = _makespan(best_sequence, processing_times)
    
    # 2. Local Search: Pairwise Swap (2-opt style)
    # Try swapping adjacent jobs and non-adjacent jobs to improve makespan.
    # To keep it fast, we'll do a few passes of adjacent swaps and random swaps.
    
    improved = True
    max_iterations = 5000
    iterations = 0
    
    while improved and iterations < max_iterations:
        improved = False
        iterations += 1
        
        n = len(best_sequence)
        
        # Adjacent swaps
        for i in range(n - 1):
            # Swap job at i and i+1
            candidate_seq = best_sequence[:]
            candidate_seq[i], candidate_seq[i+1] = candidate_seq[i+1], candidate_seq[i]
            
            ms = _makespan(candidate_seq, processing_times)
            
            if ms < best_makespan:
                best_makespan = ms
                best_sequence = candidate_seq
                improved = True
        
        # Random swaps (to escape local minima)
        # We limit random swaps to avoid getting stuck in a bad local minimum
        # but also to keep runtime low.
        if n > 2:
            for _ in range(10):
                i, j = np.random.choice(n, 2, replace=False)
                candidate_seq = best_sequence[:]
                candidate_seq[i], candidate_seq[j] = candidate_seq[j], candidate_seq[i]
                
                ms = _makespan(candidate_seq, processing_times)
                
                if ms < best_makespan:
                    best_makespan = ms
                    best_sequence = candidate_seq
                    improved = True
    
    return list(best_sequence)
# EVOLVE-BLOCK-END
