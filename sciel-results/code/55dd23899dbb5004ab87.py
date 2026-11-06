"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from collections import deque

def _calc_makespan(order, processing_times):
    """
    Calculate makespan using NumPy for speed.
    order: list or array of job indices
    processing_times: (n_jobs, n_machines) array
    """
    n = len(order)
    if n == 0:
        return 0.0
    if isinstance(order, list):
        order_arr = np.array(order)
    else:
        order_arr = order
        
    P_seq = processing_times[order_arr]
    M = P_seq.shape[1]
    if M == 0:
        return 0.0
    
    C = np.empty((n, M), dtype=np.float64)
    C[:, 0] = np.cumsum(P_seq[:, 0])
    
    for m in range(1, M):
        col_prev = C[:, m-1]
        col_curr = np.empty(n)
        col_curr[0] = col_prev[0] + P_seq[0, m]
        # Tight loop for the recurrence
        for j in range(1, n):
            # C[j, m] = max(C[j, m-1], C[j-1, m]) + P[j, m]
            v1 = col_prev[j]
            v2 = col_curr[j-1]
            if v2 > v1:
                v1 = v2
            col_curr[j] = v1 + P_seq[j, m]
        C[:, m] = col_curr
        
    return float(C[-1, -1])

def _tabu_search(order, processing_times, time_limit):
    """
    Perform Tabu Search with insertion moves.
    """
    n = len(order)
    if n <= 1:
        return order, _calc_makespan(order, processing_times)
    
    current_order = order[:]
    current_ms = _calc_makespan(current_order, processing_times)
    best_order = current_order[:]
    best_ms = current_ms
    
    # Tabu parameters
    tabu_tenure = max(5, int(np.sqrt(n)))
    tabu_list = deque()
    
    start_time = time.time()
    
    while time.time() < start_time + time_limit:
        # Generate neighbors
        # To avoid O(n^2) full evaluation if n is large, we can limit moves
        # But for n < 100, O(n^2) is fine.
        # Let's assume n is manageable. If n > 200, we might need to sample.
        
        best_neighbor_ms = float('inf')
        best_neighbor_order = None
        best_move = None
        
        # Sample moves if n is large to save time
        if n > 100:
            # Sample 10% of jobs to move
            jobs_to_try = random.sample(range(n), max(5, n // 10))
        else:
            jobs_to_try = range(n)
            
        for i in jobs_to_try:
            job = current_order[i]
            # Remove job
            remaining = current_order[:i] + current_order[i+1:]
            
            # Try inserting at different positions
            # Limit positions to nearby ones or all?
            # All positions is O(n) per job.
            for j in range(len(remaining) + 1):
                # Skip if same position
                if j == i or (j == i - 1):
                    continue
                
                # Check tabu status
                # Move is defined by (job, target_pos_j)
                # We forbid moving job to position j if it was moved to j recently
                is_tabu = False
                if tabu_list:
                    # Check if (job, j) is in tabu list
                    # This is O(len(tabu_list)) which is small
                    for (t_job, t_pos) in tabu_list:
                        if t_job == job and t_pos == j:
                            is_tabu = True
                            break
                
                # Calculate new makespan
                new_order = remaining[:j] + [job] + remaining[j:]
                new_ms = _calc_makespan(new_order, processing_times)
                
                # Aspiration criterion: accept if better than best found
                if new_ms < best_ms or not is_tabu:
                    if new_ms < best_neighbor_ms:
                        best_neighbor_ms = new_ms
                        best_neighbor_order = new_order
                        best_move = (job, j)
            
        if best_neighbor_order is None:
            break
            
        # Update tabu list
        tabu_list.append(best_move)
        if len(tabu_list) > tabu_tenure:
            tabu_list.popleft()
            
        # Accept move
        current_order = best_neighbor_order
        current_ms = best_neighbor_ms
        
        if current_ms < best_ms:
            best_ms = current_ms
            best_order = current_order[:]
            
    return best_order, best_ms

def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan.
    """
    start_time = time.time()
    deadline = start_time + 118.0  # 2s safety margin for 120s limit
    
    n_jobs = processing_times.shape[0]
    n_machines = processing_times.shape[1]
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
        
    # 1. Initial Solution: Sort by sum of processing times (Descending)
    # This is a fast, reasonable heuristic (similar to LPT)
    job_sums = processing_times.sum(axis=1)
    initial_order = list(np.argsort(-job_sums))
    
    # 2. Tabu Search
    # Allocate most time to Tabu Search
    ts_time_limit = deadline - time.time()
    
    best_order, best_ms = _tabu_search(initial_order, processing_times, ts_time_limit)
    
    # 3. Optional: Quick Local Search Polish (Best Improvement)
    # If time remains, do a few iterations of best-improvement local search
    if time.time() < deadline - 1.0:
        # Simple local search
        improved = True
        while improved and time.time() < deadline - 0.5:
            improved = False
            n = len(best_order)
            # Try all swaps or insertions? Insertions are better.
            # To keep it fast, try a few random insertions
            for _ in range(10): # Limit checks
                if time.time() > deadline - 0.5:
                    break
                i = random.randint(0, n - 1)
                j = random.randint(0, n - 1)
                if i == j or i == j - 1:
                    continue
                job = best_order[i]
                remaining = best_order[:i] + best_order[i+1:]
                if j > i:
                    j -= 1
                new_order = remaining[:j] + [job] + remaining[j:]
                new_ms = _calc_makespan(new_order, processing_times)
                if new_ms < best_ms:
                    best_order = new_order
                    best_ms = new_ms
                    improved = True
                    break
                    
    return best_order

# EVOLVE-BLOCK-END
