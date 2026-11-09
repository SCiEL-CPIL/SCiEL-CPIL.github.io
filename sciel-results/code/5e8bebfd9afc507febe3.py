"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _makespan(processing_times, order):
    """Compute makespan for a given job order efficiently using the recurrence."""
    n_jobs = len(order)
    if n_jobs == 0:
        return 0.0
    n_machines = processing_times.shape[1]
    
    # C[i][j] = max(C[i-1][j], C[i][j-1]) + p[i][j]
    # We only need the previous row and current row.
    c_prev = np.zeros(n_machines, dtype=np.float64)
    
    for i in range(n_jobs):
        job_idx = order[i]
        p_row = processing_times[job_idx]
        c_curr = np.zeros(n_machines, dtype=np.float64)
        # c_curr[j] = max(c_prev[j], c_curr[j-1]) + p_row[j]
        # This is a sequential dependency along j, so we can't fully vectorize easily
        # without a scan, but for small M it's fast in Python or we can use a loop.
        # Let's use a simple loop for clarity and correctness, it's O(N*M).
        for j in range(n_machines):
            val = p_row[j]
            if j == 0:
                c_curr[j] = c_prev[j] + val
            else:
                c_curr[j] = max(c_prev[j], c_curr[j-1]) + val
        c_prev = c_curr
        
    return c_prev[-1]


def _neh_rule(processing_times):
    """NEH heuristic: insert jobs one by one at the best position."""
    n_jobs = processing_times.shape[0]
    n_machines = processing_times.shape[1]
    
    # Sort jobs by decreasing total processing time
    totals = processing_times.sum(axis=1)
    order_indices = np.argsort(-totals)
    
    perm = []
    for job_idx in order_indices:
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(perm) + 1):
            new_perm = perm[:pos] + [job_idx] + perm[pos:]
            ms = _makespan(processing_times, new_perm)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        perm.insert(best_pos, job_idx)
        
    return perm


def _local_search(processing_times, initial_order, max_time=10.0):
    """First-improvement local search using adjacent swaps (2-opt)."""
    start_time = time.time()
    n_jobs = len(initial_order)
    if n_jobs <= 1:
        return initial_order
    
    current_order = list(initial_order)
    current_ms = _makespan(processing_times, current_order)
    
    improved = True
    while improved:
        improved = False
        for i in range(n_jobs - 1):
            if time.time() - start_time > max_time:
                return current_order
            
            # Swap i and i+1
            new_order = current_order.copy()
            new_order[i], new_order[i+1] = new_order[i+1], new_order[i]
            new_ms = _makespan(processing_times, new_order)
            
            if new_ms < current_ms:
                current_order = new_order
                current_ms = new_ms
                improved = True
                # Restart scan from beginning for first-improvement
                break
                
    return current_order


def order_jobs(processing_times):
    """
    Main entry point. Uses NEH heuristic followed by local search.
    Includes a time guard to ensure completion within limits.
    """
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s limit
    
    n_jobs = processing_times.shape[0]
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # 1. Generate initial solution using NEH
    initial_order = _neh_rule(processing_times)
    current_best_order = initial_order
    current_best_ms = _makespan(processing_times, current_best_order)
    
    # 2. Perform local search
    if time.time() - start_time < deadline - 5.0:
        ls_time_budget = min(10.0, deadline - (time.time() - start_time) - 5.0)
        if ls_time_budget > 0.1:
            improved_order = _local_search(processing_times, current_best_order, max_time=ls_time_budget)
            improved_ms = _makespan(processing_times, improved_order)
            if improved_ms < current_best_ms:
                current_best_order = improved_order
                current_best_ms = improved_ms
    
    # 3. If time remains, try a few random restarts with NEH + LS
    #    This is optional and only if we have significant time left.
    #    Given the strict time limit and the effectiveness of NEH+LS, 
    #    we might skip this to be safe, or do a very limited version.
    
    return current_best_order

# EVOLVE-BLOCK-END
