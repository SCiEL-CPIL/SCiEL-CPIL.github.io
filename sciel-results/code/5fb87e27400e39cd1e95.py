"""Improved program for flow-shop permutation construction using NEH heuristic with time budget and local search."""
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

def _local_search(order, P, deadline=None, max_passes=5):
    """
    2-Opt style local search for flow shop.
    Tries to improve the makespan by swapping adjacent jobs or moving a job to a different position.
    Uses a simple first-improvement strategy within a time budget.
    """
    n_jobs = len(order)
    if n_jobs < 2:
        return order
    
    current_order = list(order)
    current_ms = _makespan(current_order, P)
    
    # We will do multiple passes until no improvement is found or deadline is hit
    for _pass in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        for i in range(n_jobs - 1):
            if deadline is not None and time.time() > deadline:
                break
            
            # Try swapping adjacent jobs i and i+1
            new_order = current_order.copy()
            new_order[i], new_order[i+1] = new_order[i+1], new_order[i]
            
            new_ms = _makespan(new_order, P)
            if new_ms < current_ms:
                current_order = new_order
                current_ms = new_ms
                improved = True
        
        # If no improvement was found in this pass, we could try other moves or break
        # For simplicity, let's stick to adjacent swaps for now, but maybe add a second type of move?
        # Let's add a "move" operation: pick a job and move it to a different position.
        # This is more expensive but might escape local minima better.
        
        # However, adjacent swaps are O(n * n * m) per pass.
        # Let's stick to adjacent swaps for efficiency, but maybe do a few more passes if time permits.
        
        if not improved:
            # No improvement from adjacent swaps. 
            # We could try a more expensive move, but let's be conservative.
            # Actually, let's try a simple "re-insert" of a random job? No, let's keep it deterministic.
            # Let's try moving each job to the best position? That's NEH again.
            # Let's just break if no improvement from adjacent swaps.
            break
            
    return current_order

def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan on a flow-shop instance.
    Uses the NEH heuristic followed by a local search improvement.
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
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Apply NEH
    permutation = _neh_heuristic(P, deadline)
    
    # If permutation is incomplete due to timeout, complete it by appending remaining jobs
    if len(permutation) < n_jobs:
        remaining_jobs = [j for j in range(n_jobs) if j not in permutation]
        # Sort remaining by total processing time descending and append
        totals = P.sum(axis=1)
        remaining_jobs.sort(key=lambda x: -totals[x])
        permutation.extend(remaining_jobs)
    
    # Apply local search if there is time remaining
    # Check if we have at least 5 seconds left before the hard deadline
    if time.time() < deadline - 5.0:
        # Adjust deadline for local search to leave some buffer
        ls_deadline = deadline - 2.0
        permutation = _local_search(permutation, P, deadline=ls_deadline)
    
    return list(permutation)

# EVOLVE-BLOCK-END
