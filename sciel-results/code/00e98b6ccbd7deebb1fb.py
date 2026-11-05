import numpy as np
import time

def order_jobs(processing_times):
    """
    Constructs a job permutation minimizing makespan on a flow-shop instance.
    Uses NEH heuristic for construction and local search (pairwise swaps) for refinement.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin: 110s out of 120s
    
    n, m = processing_times.shape
    
    # 1. Constructive Heuristic: NEH (Nawaz, Enscore, Ham)
    # Sort jobs by total processing time in descending order
    total_pt = processing_times.sum(axis=1)
    job_order = np.argsort(-total_pt)
    
    # Build partial sequence
    seq = []
    for job_idx in job_order:
        # Try inserting job_idx into each position in seq
        best_pos = 0
        best_makespan = np.inf
        
        # If seq is empty, just add
        if not seq:
            seq.append(job_idx)
            continue
            
        for pos in range(len(seq) + 1):
            # Create candidate sequence
            cand_seq = seq[:pos] + [job_idx] + seq[pos:]
            
            # Calculate makespan for candidate sequence
            ms = _calc_makespan(cand_seq, processing_times)
            
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
        
        # Insert at best position
        seq = seq[:best_pos] + [job_idx] + seq[best_pos:]
        
        # Time check
        if time.time() > deadline:
            break

    # Convert to numpy array for easier manipulation in local search
    current_perm = np.array(seq)
    best_perm = current_perm.copy()
    best_ms = _calc_makespan(best_perm, processing_times)
    
    # 2. Local Search: Pairwise Swap
    # We perform iterations of local search until no improvement or time limit
    while time.time() < deadline:
        improved = False
        n_curr = len(current_perm)
        
        # To speed up, we can limit the search or use a simpler neighborhood
        # Standard 2-opt style swap
        for i in range(n_curr):
            if time.time() > deadline:
                break
            for j in range(i + 1, n_curr):
                # Swap i and j
                current_perm[i], current_perm[j] = current_perm[j], current_perm[i]
                new_ms = _calc_makespan(current_perm, processing_times)
                
                if new_ms < best_ms:
                    best_ms = new_ms
                    best_perm = current_perm.copy()
                    improved = True
                else:
                    # Revert
                    current_perm[i], current_perm[j] = current_perm[j], current_perm[i]
        
        if not improved:
            break
            
    return list(best_perm)

def _calc_makespan(perm, processing_times):
    """
    Efficiently calculates the makespan for a given permutation.
    perm: array of job indices
    processing_times: (n, m) array
    """
    n, m = processing_times.shape
    if n == 0 or m == 0:
        return 0.0
        
    # DP table for completion times
    # C[i][k] is the completion time of job perm[i] on machine k
    # We can optimize memory by only keeping the previous row and current row
    # C[i][k] = max(C[i-1][k], C[i][k-1]) + P[perm[i]][k]
    
    prev_row = np.zeros(m)
    curr_row = np.zeros(m)
    
    for i in range(n):
        job_idx = perm[i]
        p = processing_times[job_idx]
        
        # First machine
        curr_row[0] = prev_row[0] + p[0]
        
        for k in range(1, m):
            # C[i][k] = max(C[i-1][k], C[i][k-1]) + P[i][k]
            curr_row[k] = max(prev_row[k], curr_row[k-1]) + p[k]
            
        # Update prev_row for next iteration
        prev_row = curr_row.copy()
        
    return float(prev_row[-1])
