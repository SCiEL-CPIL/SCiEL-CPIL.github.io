"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _makespan(perm, pt):
    """Compute makespan for a given permutation."""
    n = len(perm)
    if n == 0:
        return 0.0
    m = pt.shape[1]
    # completion[m] tracks completion time on machine m
    # For job in perm: for each machine k: C[k] = max(C[k-1], C[k]) + pt[job][k]
    # This is O(n*m) but sequential over machines
    comp = np.zeros(m)
    for j in perm:
        for k in range(m):
            if k == 0:
                comp[0] += pt[j, 0]
            else:
                comp[k] = max(comp[k - 1], comp[k]) + pt[j, k]
    return comp[-1]


def _makespan_vectorized(perm, pt):
    """Vectorized makespan computation."""
    n = len(perm)
    if n == 0:
        return 0.0
    m = pt.shape[1]
    # For each machine, we can compute completion times
    # C[j][k] = max(C[j-1][k], C[j][k-1]) + pt[j][k]
    # This requires sequential computation over both j and k
    # But we can at least vectorize over the inner loop partially
    comp = np.zeros(m)
    for j in perm:
        row = pt[j]
        # comp[k] = max(comp[k-1], comp[k]) + row[k]
        # This is inherently sequential, but let's do it in a loop
        new_comp = np.empty(m)
        new_comp[0] = comp[0] + row[0]
        for k in range(1, m):
            new_comp[k] = max(new_comp[k - 1], comp[k]) + row[k]
        comp = new_comp
    return comp[-1]


def _delta_swap(perm, i, j, pt):
    """
    Compute the change in makespan when swapping jobs at positions i and j in perm.
    i < j. Returns the new makespan.
    
    For a swap of adjacent jobs at positions i and i+1, the delta can be computed
    in O(m). For non-adjacent swaps, it's more complex.
    
    Let's handle adjacent swaps first for efficiency.
    """
    if j != i + 1:
        # Non-adjacent swap: just recompute
        new_perm = list(perm)
        new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
        return _makespan_vectorized(new_perm, pt)
    
    # Adjacent swap: positions i and i+1
    # Let a = perm[i], b = perm[i+1]
    # Before: ... a, b, ...
    # After: ... b, a, ...
    
    n = len(perm)
    m = pt.shape[1]
    
    # Compute completion times up to position i-1
    comp_before = np.zeros(m)
    for pos in range(i):
        j_idx = perm[pos]
        row = pt[j_idx]
        new_comp = np.empty(m)
        new_comp[0] = comp_before[0] + row[0]
        for k in range(1, m):
            new_comp[k] = max(new_comp[k - 1], comp_before[k]) + row[k]
        comp_before = new_comp
    
    # Now compute makespan with a, b order
    # comp_after_ab = process a then b starting from comp_before
    def process_job(comp_start, job_idx):
        row = pt[job_idx]
        new_comp = np.empty(m)
        new_comp[0] = comp_start[0] + row[0]
        for k in range(1, m):
            new_comp[k] = max(new_comp[k - 1], comp_start[k]) + row[k]
        return new_comp
    
    comp_a = process_job(comp_before, perm[i])
    comp_ab = process_job(comp_a, perm[i + 1])
    
    # Process remaining jobs after i+1
    comp_end_ab = comp_ab.copy()
    for pos in range(i + 2, n):
        j_idx = perm[pos]
        comp_end_ab = process_job(comp_end_ab, j_idx)
    makespan_ab = comp_end_ab[-1]
    
    # Now compute makespan with b, a order
    comp_b = process_job(comp_before, perm[i + 1])
    comp_ba = process_job(comp_b, perm[i])
    
    comp_end_ba = comp_ba.copy()
    for pos in range(i + 2, n):
        j_idx = perm[pos]
        comp_end_ba = process_job(comp_end_ba, j_idx)
    makespan_ba = comp_end_ba[-1]
    
    return makespan_ba


def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan on a flow-shop instance.
    Uses NEH heuristic for initial solution, then local search with 2-opt moves.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    pt = np.asarray(processing_times, dtype=np.float64)
    n_jobs, n_machines = pt.shape
    
    if n_jobs <= 1:
        return list(range(n_jobs))
    
    # Step 1: NEH heuristic for initial solution
    # Sort jobs by total processing time in descending order
    totals = pt.sum(axis=1)
    order = np.argsort(-totals)
    
    # Build NEH solution
    perm = []
    for j in order:
        # Find best insertion position
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(perm) + 1):
            new_perm = perm[:pos] + [j] + perm[pos:]
            ms = _makespan_vectorized(new_perm, pt)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        perm.insert(best_pos, j)
    
    current_ms = _makespan_vectorized(perm, pt)
    best_perm = list(perm)
    best_ms = current_ms
    
    # Step 2: Local search with adjacent swaps (2-opt)
    improved = True
    while improved:
        improved = False
        for i in range(n_jobs - 1):
            if time.time() > deadline:
                break
            # Try swapping adjacent jobs at positions i and i+1
            new_ms = _delta_swap(perm, i, i + 1, pt)
            if new_ms < current_ms - 1e-9:
                # Accept the swap
                perm[i], perm[i + 1] = perm[i + 1], perm[i]
                current_ms = new_ms
                if current_ms < best_ms:
                    best_ms = current_ms
                    best_perm = list(perm)
                improved = True
                # Restart scan from beginning after improvement
                break
        if time.time() > deadline:
            break
    
    # Step 3: Try some random restarts if time permits
    rng = np.random.RandomState(42)
    attempts = 0
    while time.time() < deadline - 5.0:
        attempts += 1
        # Generate a random perturbation of the best solution
        perm = list(best_perm)
        # Do a few random swaps
        for _ in range(min(n_jobs, 10)):
            i, j = rng.randint(0, n_jobs, 2)
            perm[i], perm[j] = perm[j], perm[i]
        
        current_ms = _makespan_vectorized(perm, pt)
        
        # Local search from this perturbed solution
        improved = True
        while improved:
            improved = False
            for i in range(n_jobs - 1):
                if time.time() > deadline - 5.0:
                    break
                new_ms = _delta_swap(perm, i, i + 1, pt)
                if new_ms < current_ms - 1e-9:
                    perm[i], perm[i + 1] = perm[i + 1], perm[i]
                    current_ms = new_ms
                    if current_ms < best_ms:
                        best_ms = current_ms
                        best_perm = list(perm)
                    improved = True
                    break
            if time.time() > deadline - 5.0:
                break
    
    return best_perm
# EVOLVE-BLOCK-END
