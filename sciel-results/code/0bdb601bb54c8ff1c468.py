# EVOLVE-BLOCK-START
import numpy as np
import time


def _makespan(perm, p):
    """Compute makespan for a given permutation."""
    if len(perm) == 0:
        return 0.0
    n = len(perm)
    m = p.shape[1]
    # C[j, k] = completion time of job j on machine k
    # We can compute this efficiently
    # C[j, 0] = sum of p[perm[j], 0] for j' <= j
    # C[j, k] = max(C[j, k-1], C[j-1, k]) + p[perm[j], k]
    
    # For efficiency, we track the cumulative sum on each machine
    # Actually, let's use the standard flow shop formula
    # C[j][k] = max(C[j][k-1], C[j-1][k]) + p[perm[j]][k]
    
    # This is O(n*m) which is fine
    prev_row = np.zeros(m)
    curr_row = np.zeros(m)
    
    for j in range(n):
        job = perm[j]
        curr_row[0] = prev_row[0] + p[job, 0]
        for k in range(1, m):
            curr_row[k] = max(curr_row[k-1], prev_row[k]) + p[job, k]
        prev_row = curr_row.copy()
    
    return prev_row[-1]


def _insertion_cost(perm, pos, job, p):
    """
    Compute the change in makespan if we insert job at position pos in perm.
    Returns the new makespan.
    """
    # This is expensive, so we'll use it sparingly
    new_perm = list(perm[:pos]) + [job] + list(perm[pos:])
    return _makespan(new_perm, p)


def order_jobs(processing_times):
    n, m = processing_times.shape
    start_time = time.time()
    deadline = start_time + 110  # 10 seconds safety margin before 120s limit
    
    if n <= 1:
        return list(range(n))
    
    # Step 1: Generate initial solution using NEH-like heuristic
    # Sort jobs by total processing time in descending order
    total_pt = processing_times.sum(axis=1)
    init_perm = list(np.argsort(-total_pt))
    best_perm = list(init_perm)
    best_ms = _makespan(best_perm, processing_times)
    
    # Step 2: Iterated Local Search with Or-opt and 2-opt moves
    # We'll use a simple ILS approach
    
    current_perm = list(best_perm)
    current_ms = best_ms
    
    # Precompute for faster evaluation
    # For each job and machine, we have the processing time
    
    # Local search: try all single-job insertions (Or-opt with k=1)
    def local_search(perm, ms, p, n, deadline):
        improved = True
        while improved:
            improved = False
            for i in range(n):
                if time.time() > deadline:
                    return perm, ms
                job = perm[i]
                # Remove job from position i
                remaining = [x for x in perm if x != job]
                # Try inserting at each position
                for j in range(len(remaining)):
                    if time.time() > deadline:
                        return perm, ms
                    new_perm = remaining[:j] + [job] + remaining[j:]
                    new_ms = _makespan(new_perm, p)
                    if new_ms < ms - 1e-9:
                        perm = new_perm
                        ms = new_ms
                        improved = True
                        break
            if time.time() > deadline:
                break
        return perm, ms
    
    # Run local search from initial solution
    current_perm, current_ms = local_search(current_perm, current_ms, processing_times, n, deadline)
    
    if current_ms < best_ms:
        best_ms = current_ms
        best_perm = list(current_perm)
    
    # Iterated Local Search: perturb and local search
    while time.time() < deadline:
        # Perturbation: random swap of two jobs
        perm = list(best_perm)
        i, j = np.random.choice(n, 2, replace=False)
        perm[i], perm[j] = perm[j], perm[i]
        
        # Local search
        perm, ms = local_search(perm, _makespan(perm, processing_times), processing_times, n, deadline)
        
        if ms < best_ms:
            best_ms = ms
            best_perm = list(perm)
        
        # If we've used most of our time, break
        if time.time() > deadline - 5:
            break
    
    return best_perm
# EVOLVE-BLOCK-END
