"""Improved program for flow-shop permutation construction using NEH heuristic with time budget."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _makespan(order, P):
    """Compute makespan for a given job order on processing_times matrix P."""
    if len(order) == 0:
        return 0.0
    n_jobs = len(order)
    n_machines = P.shape[1]
    
    if n_machines == 0:
        return 0.0
        
    # Initialize for the first job in the sequence
    first_job = order[0]
    # For the first job, C(0, m) = sum(P[0, 0..m])
    curr_row = np.cumsum(P[first_job])
    
    for idx in range(1, n_jobs):
        j = order[idx]
        # curr_row was C(idx-1, m) for all m. Let's call it prev_row.
        prev_row = curr_row
        # Compute new curr_row for job j
        # C(j, 0) = C(j-1, 0) + P(j, 0)
        # C(j, m) = max(C(j-1, m), C(j, m-1)) + P(j, m)
        new_row = np.empty(n_machines)
        new_row[0] = prev_row[0] + P[j, 0]
        if n_machines > 1:
            for m in range(1, n_machines):
                new_row[m] = max(prev_row[m], new_row[m-1]) + P[j, m]
        curr_row = new_row
    
    return curr_row[-1]

def _neh_heuristic(P, deadline=None):
    """
    NEH heuristic for flow shop scheduling.
    1. Sort jobs in descending order of total processing time.
    2. Insert each job into the best position (minimizing makespan) in the current partial sequence.
    If deadline is provided, stops early if time runs out, returning the current best sequence.
    """
    n_jobs, n_machines = P.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # Step 1: Sort by total processing time descending
    totals = P.sum(axis=1)
    sorted_indices = np.argsort(-totals)
    
    # Start with the first job
    current_seq = [int(sorted_indices[0])]
    
    # Insert remaining jobs one by one
    for k in range(1, n_jobs):
        # Check deadline
        if deadline is not None:
            if time.time() > deadline:
                break
        
        job = int(sorted_indices[k])
        best_pos = -1
        best_makespan = float('inf')
        
        # Try all possible insertion positions (0 to len(current_seq))
        for pos in range(len(current_seq) + 1):
            # Check deadline inside inner loop for large n_jobs
            if deadline is not None and pos % 10 == 0:
                if time.time() > deadline:
                    break
            
            # Create new sequence
            new_seq = current_seq[:pos] + [job] + current_seq[pos:]
            # Calculate makespan
            ms = _makespan(new_seq, P)
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
        
        # If deadline was hit inside inner loop, best_pos might be -1 or partial
        # If best_pos is -1, we should probably just append or break
        if best_pos == -1:
            # If we couldn't find a valid position due to timeout, just append to end
            current_seq.append(job)
        else:
            # Insert at best position
            current_seq.insert(best_pos, job)
        
    return current_seq

def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan on a flow-shop instance.
    Uses the NEH heuristic, which is a strong constructive heuristic for FSP.
    Includes a time budget to ensure it finishes within the limit.
    """
    P = np.asarray(processing_times)
    if P.size == 0:
        return []
        
    n_jobs = P.shape[0]
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # Set a deadline with a safety margin (e.g., 10s before the 120s limit, or less if needed)
    # The task says 120s limit. Let's use 110s to be safe.
    # However, NEH is O(n^2 * m). For typical n (e.g., 100-1000), this should be fast.
    # If n is very large, we might need to be more careful.
    # Let's set deadline to current time + 110 seconds.
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Apply NEH
    permutation = _neh_heuristic(P, deadline)
    
    # If permutation is incomplete due to timeout, complete it by appending remaining jobs
    # This ensures we return a valid permutation of all jobs.
    if len(permutation) < n_jobs:
        remaining_jobs = [j for j in range(n_jobs) if j not in permutation]
        # Sort remaining by total processing time descending and append
        totals = P.sum(axis=1)
        remaining_jobs.sort(key=lambda x: -totals[x])
        permutation.extend(remaining_jobs)
    
    return list(permutation)

# EVOLVE-BLOCK-END
