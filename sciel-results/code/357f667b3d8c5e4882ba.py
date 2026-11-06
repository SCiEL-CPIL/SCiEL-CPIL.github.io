"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _makespan(perm, pt):
    """Compute makespan for a given permutation."""
    n_jobs, n_machines = pt.shape
    if n_jobs == 0:
        return 0.0
    # Compute completion times using the flow-shop recurrence
    # C[j][m] = max(C[j][m-1], C[j-1][m]) + pt[perm[j]][m]
    # We only need the last machine's completion times to get makespan
    # But we need the full matrix for correctness
    # Optimized: track only the previous row and current row
    prev_row = np.zeros(n_machines)
    for j in range(n_jobs):
        job = perm[j]
        curr_row = np.empty(n_machines)
        for m in range(n_machines):
            if m == 0:
                curr_row[m] = prev_row[m] + pt[job, m] if j > 0 else pt[job, m]
            else:
                curr_row[m] = max(curr_row[m-1], prev_row[m]) + pt[job, m]
        prev_row = curr_row
    return prev_row[-1]


def _makespan_vectorized(perm, pt):
    """Vectorized makespan computation using cumulative operations."""
    n_jobs, n_machines = pt.shape
    if n_jobs == 0:
        return 0.0
    # Extract processing times in the order of the permutation
    p = pt[perm]  # shape (n_jobs, n_machines)
    
    # Use the recurrence: C[j][m] = max(C[j][m-1], C[j-1][m]) + p[j][m]
    # This is inherently sequential along both dimensions, but we can use
    # a loop over jobs with vectorized ops over machines
    prev = np.zeros(n_machines)
    for j in range(n_jobs):
        # curr[m] = max(curr[m-1], prev[m]) + p[j][m]
        # This requires sequential computation along m
        curr = np.empty(n_machines)
        for m in range(n_machines):
            if m == 0:
                curr[m] = prev[m] + p[j, m]
            else:
                curr[m] = max(curr[m-1], prev[m]) + p[j, m]
        prev = curr
    return prev[-1]


def _neh_construction(pt):
    """NEH (Nawaz, Enscore, Ham) heuristic for initial construction."""
    n_jobs, n_machines = pt.shape
    if n_jobs <= 1:
        return list(range(n_jobs))
    
    # Sort jobs by decreasing total processing time
    total_times = pt.sum(axis=1)
    order = list(np.argsort(-total_times))
    
    # Build the sequence using NEH rule: insert each job at the best position
    seq = [order[0]]
    for j in order[1:]:
        best_pos = 0
        best_makespan = float('inf')
        for pos in range(len(seq) + 1):
            # Insert job j at position pos
            new_seq = seq[:pos] + [j] + seq[pos:]
            ms = _makespan_vectorized(new_seq, pt)
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
        seq = seq[:best_pos] + [j] + seq[best_pos:]
    
    return seq


def _insertion_local_search(perm, pt, max_passes=50, deadline=None):
    """Local search with insertion moves (I1 neighborhood)."""
    n_jobs = len(perm)
    if n_jobs <= 1:
        return perm
    
    current_perm = list(perm)
    current_ms = _makespan_vectorized(current_perm, pt)
    best_perm = list(current_perm)
    best_ms = current_ms
    
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline - 2.0:
            break
        
        improved = False
        for i in range(n_jobs):
            if deadline is not None and time.time() > deadline - 2.0:
                break
            
            job = current_perm[i]
            # Remove job from position i
            remaining = current_perm[:i] + current_perm[i+1:]
            
            # Try inserting at each position in remaining
            for pos in range(len(remaining) + 1):
                if pos == i:
                    continue
                new_perm = remaining[:pos] + [job] + remaining[pos:]
                new_ms = _makespan_vectorized(new_perm, pt)
                if new_ms < current_ms:
                    current_perm = new_perm
                    current_ms = new_ms
                    improved = True
                    if new_ms < best_ms:
                        best_ms = new_ms
                        best_perm = list(current_perm)
                    break  # Accept first improvement and restart scan
    
        if not improved:
            break
    
    return best_perm


def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan using:
    1. NEH heuristic for initial construction.
    2. Insertion-based local search to refine.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    pt = np.asarray(processing_times, dtype=np.float64)
    n_jobs, n_machines = pt.shape
    
    if n_jobs <= 1:
        return list(range(n_jobs))
    
    # Stage 1: NEH construction
    perm = _neh_construction(pt)
    current_ms = _makespan_vectorized(perm, pt)
    
    # Stage 2: Local search with deadline guard
    perm = _insertion_local_search(perm, pt, max_passes=200, deadline=deadline)
    
    return perm
# EVOLVE-BLOCK-END
