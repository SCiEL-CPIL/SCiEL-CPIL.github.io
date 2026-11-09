# EVOLVE-BLOCK-START
import numpy as np
import time


def _makespan(pt, perm):
    """Compute makespan for a given permutation on flow-shop."""
    n, m = pt.shape
    if n == 0:
        return 0.0
    # Build completion time matrix row by row
    # C[j][k] = completion time of job perm[j] on machine k
    # Use vectorized-ish approach: for each machine, compute sequentially
    # But for small n,m this is fine with a loop
    # We'll use a numpy array for speed
    C = np.zeros((n, m), dtype=np.float64)
    for j in range(n):
        job = perm[j]
        for k in range(m):
            if j == 0:
                C[j, k] = C[j, k-1] + pt[job, k] if k > 0 else pt[job, k]
            else:
                C[j, k] = max(C[j-1, k], C[j, k-1] if k > 0 else 0) + pt[job, k]
    return C[-1, -1]


def _makespan_fast(pt, perm):
    """Faster makespan computation using cumulative sums and max operations."""
    n = len(perm)
    m = pt.shape[1]
    if n == 0:
        return 0.0
    
    # Extract the processing times in the order of the permutation
    p = pt[perm, :]  # shape (n, m)
    
    # Compute completion times
    # C[j, k] = max(C[j-1, k], C[j, k-1]) + p[j, k]
    # This is the standard flow shop recurrence
    
    C = np.zeros((n, m), dtype=np.float64)
    for j in range(n):
        for k in range(m):
            top = C[j-1, k] if j > 0 else 0.0
            left = C[j, k-1] if k > 0 else 0.0
            C[j, k] = max(top, left) + p[j, k]
    
    return C[-1, -1]


def _neh_construct(pt):
    """NEH heuristic for flow shop."""
    n, m = pt.shape
    if n == 0:
        return []
    
    # Sort jobs by total processing time in descending order
    totals = pt.sum(axis=1)
    order = list(np.argsort(-totals))
    
    # Insert jobs one by one into the best position
    perm = []
    for job in order:
        # Try inserting at each position
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(perm) + 1):
            trial = perm[:pos] + [job] + perm[pos:]
            ms = _makespan_fast(pt, trial)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        perm = perm[:best_pos] + [job] + perm[best_pos:]
    
    return perm


def _insertion_local_search(pt, perm, max_iters=100):
    """Insertion-based local search: try moving each job to every other position."""
    n = len(perm)
    if n <= 1:
        return perm
    
    current = list(perm)
    current_ms = _makespan_fast(pt, current)
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        for i in range(n):
            job = current[i]
            for j in range(n):
                if i == j:
                    continue
                # Move job from position i to position j
                trial = current[:i] + current[i+1:]
                # Insert at position j (adjust if j > i)
                insert_pos = j if j < i else j
                trial = trial[:insert_pos] + [job] + trial[insert_pos:]
                ms = _makespan_fast(pt, trial)
                if ms < current_ms - 1e-9:
                    current = trial
                    current_ms = ms
                    improved = True
                    break
            if improved:
                break
    return current


def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan on flow-shop instances.
    
    Uses NEH construction followed by insertion-based local search,
    with multiple diverse starting points and a time budget guard.
    """
    pt = np.asarray(processing_times, dtype=np.float64)
    n, m = pt.shape
    
    if n <= 1:
        return list(range(n))
    
    deadline = 115.0  # Safety margin before 120s limit
    start_time = time.time()
    
    # Generate diverse initial solutions
    initial_perms = []
    
    # 1. NEH heuristic
    initial_perms.append(_neh_construct(pt))
    
    # 2. SPT (Shortest Processing Time) on first machine
    spt_order = list(np.argsort(pt[:, 0]))
    initial_perms.append(spt_order)
    
    # 3. LPT (Longest Processing Time) on last machine
    lpt_order = list(np.argsort(-pt[:, -1]))
    initial_perms.append(lpt_order)
    
    # 4. Total processing time descending (original seed)
    total_order = list(np.argsort(-pt.sum(axis=1)))
    initial_perms.append(total_order)
    
    # 5. Random permutations (a few)
    rng = np.random.default_rng(42)
    for _ in range(3):
        initial_perms.append(list(rng.permutation(n)))
    
    # Evaluate and improve each initial solution
    best_perm = initial_perms[0]
    best_ms = _makespan_fast(pt, best_perm)
    
    for perm in initial_perms[1:]:
        if time.time() - start_time > deadline:
            break
        # Local search on this permutation
        improved = _insertion_local_search(pt, perm, max_iters=50)
        ms = _makespan_fast(pt, improved)
        if ms < best_ms:
            best_ms = ms
            best_perm = improved
    
    # Additional local search iterations on the best solution
    if time.time() - start_time < deadline:
        # Do more rounds of local search on the best
        for _ in range(5):
            if time.time() - start_time > deadline:
                break
            improved = _insertion_local_search(pt, best_perm, max_iters=30)
            ms = _makespan_fast(pt, improved)
            if ms < best_ms:
                best_ms = ms
                best_perm = improved
            else:
                break
    
    return list(best_perm)
# EVOLVE-BLOCK-END
