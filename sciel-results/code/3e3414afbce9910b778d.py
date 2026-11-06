"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _makespan(order, P):
    """
    Calculate the makespan for a given job order in a flow shop.
    P: (n_jobs, n_machines) array
    order: list of job indices
    """
    if not order:
        return 0.0
    
    n_machines = P.shape[1]
    if n_machines == 0:
        return 0.0
    
    # C[j] is the completion time of the current job on machine j
    # We track the completion time of the previous job on each machine
    prev_completion = np.zeros(n_machines)
    
    for j in order:
        # Processing time for job j on all machines
        p_j = P[j]
        
        # Completion time on machine 0
        curr_completion = prev_completion[0] + p_j[0]
        
        if n_machines == 1:
            prev_completion[0] = curr_completion
            continue
            
        # For subsequent machines
        for m in range(1, n_machines):
            curr_completion = max(curr_completion, prev_completion[m]) + p_j[m]
            prev_completion[m] = curr_completion
            
    return prev_completion[-1]

def _neh_heuristic(P):
    """
    Nawaz, Enscore, and Ham (NEH) heuristic.
    """
    n_jobs, n_machines = P.shape
    
    # Sum of processing times for each job
    sums = P.sum(axis=1)
    
    # Sort jobs by total processing time in descending order
    sorted_jobs = list(np.argsort(-sums))
    
    sequence = []
    
    for job in sorted_jobs:
        best_pos = 0
        best_ms = float('inf')
        
        # Try inserting the job at every position
        for pos in range(len(sequence) + 1):
            new_seq = sequence[:pos] + [job] + sequence[pos:]
            ms = _makespan(new_seq, P)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        
        sequence.insert(best_pos, job)
        
    return sequence

def _local_search(initial_order, P, time_limit=110.0):
    """
    Local search using insertion moves.
    """
    start_time = time.time()
    order = list(initial_order)
    current_ms = _makespan(order, P)
    n_jobs = len(order)
    
    improved = True
    while improved:
        improved = False
        # Check time limit
        if time.time() - start_time > time_limit:
            break
            
        for i in range(n_jobs):
            if time.time() - start_time > time_limit:
                break
            job = order[i]
            # Try moving job i to every position j != i
            for j in range(n_jobs):
                if i == j:
                    continue
                # Create new order by removing job at i and inserting at j
                # Note: if j > i, the index shifts
                new_order = order[:i] + order[i+1:]
                insert_idx = j if j <= i else j - 1
                new_order.insert(insert_idx, job)
                
                new_ms = _makespan(new_order, P)
                if new_ms < current_ms:
                    order = new_order
                    current_ms = new_ms
                    improved = True
                    # Restart the inner loop or continue? 
                    # Usually, we restart the outer loop to re-evaluate from scratch
                    # or continue with the updated order. 
                    # To be safe and effective, break inner and restart outer.
                    break
            if improved:
                break
                
    return order, current_ms

def order_jobs(processing_times):
    # Convert to numpy array if not already
    P = np.asarray(processing_times)
    
    if P.size == 0:
        return []
        
    # 1. Generate initial solution using NEH
    best_order = _neh_heuristic(P)
    best_ms = _makespan(best_order, P)
    
    # 2. Improve using Local Search
    # We have a 120s limit. Let's reserve 10s for safety.
    # NEH is fast, so most time goes to LS.
    improved_order, improved_ms = _local_search(best_order, P, time_limit=110.0)
    
    if improved_ms < best_ms:
        best_order = improved_order
        best_ms = improved_ms
        
    return best_order
# EVOLVE-BLOCK-END
