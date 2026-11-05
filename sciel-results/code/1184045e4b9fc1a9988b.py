"""Rewritten program for flow-shop permutation task."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _makespan(pt, perm):
    """Compute makespan for a given permutation using DP."""
    n_jobs, n_machines = pt.shape
    if n_machines == 0 or n_jobs == 0:
        return 0
    
    # Use pure Python lists for speed on small M
    prev = [0.0] * n_machines
    for j in perm:
        curr = [0.0] * n_machines
        curr[0] = prev[0] + pt[j, 0]
        for m in range(1, n_machines):
            val = prev[m] if prev[m] > curr[m-1] else curr[m-1]
            curr[m] = val + pt[j, m]
        prev = curr
    return prev[-1]

def _makespan_delta_insert(pt, perm, job, old_pos, new_pos):
    """
    Compute the makespan of the permutation after removing job from old_pos 
    and inserting it at new_pos.
    perm: current permutation (list)
    job: the job to move
    old_pos: current position of job in perm
    new_pos: target position (in the permutation BEFORE removal, 
             but we handle the shift internally)
    
    Actually, let's just compute the full makespan for correctness first,
    then optimize if needed. Given the constraints, a full recomputation 
    is O(N*M). With N up to maybe 100-200 and M up to 20, this is fast enough
    if we limit the number of evaluations.
    
    However, the previous program timed out. Let's use a faster heuristic or 
    limit the search space more aggressively.
    
    For now, let's stick to a robust but efficient local search.
    """
    # Reconstruct the new permutation
    new_perm = perm[:old_pos] + perm[old_pos+1:]
    # Adjust new_pos if it was after old_pos
    if new_pos > old_pos:
        new_pos -= 1
    new_perm = new_perm[:new_pos] + [job] + new_perm[new_pos:]
    return _makespan(pt, new_perm), new_perm

def _neh_initial(pt):
    """NEH heuristic initial solution."""
    n_jobs, n_machines = pt.shape
    totals = pt.sum(axis=1)
    # Sort by descending total processing time
    order = list(np.argsort(-totals))
    
    # Refine with insertion
    current = []
    for job in order:
        best_pos = 0
        best_ms = float('inf')
        # Try inserting job at each position in current
        for pos in range(len(current) + 1):
            test_perm = current[:pos] + [job] + current[pos:]
            ms = _makespan(pt, test_perm)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        current = current[:best_pos] + [job] + current[best_pos:]
        
    return current

def _palmer_initial(pt):
    """Palmer heuristic (WSPT) initial solution."""
    n_jobs, n_machines = pt.shape
    # Palmer's priority index: sum of (n-m-1)*t[j,m] for m=0..n-1
    # Actually, it's sum_{m=1}^{M} (M - m + 1) * t_{jm}
    # Let's use the standard definition:
    # PI_j = sum_{m=1}^{M} (M - m + 1) * t_{jm}
    # But in 0-indexed: sum_{m=0}^{M-1} (M - m) * t_{jm}
    weights = np.arange(n_machines, 0, -1)
    priorities = pt.dot(weights)
    order = list(np.argsort(-priorities))
    
    # Refine with insertion
    current = []
    for job in order:
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(current) + 1):
            test_perm = current[:pos] + [job] + current[pos:]
            ms = _makespan(pt, test_perm)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        current = current[:best_pos] + [job] + current[best_pos:]
        
    return current

def _lpt_initial(pt):
    """LPT (Longest Processing Time) on first machine."""
    n_jobs, n_machines = pt.shape
    order = list(np.argsort(-pt[:, 0]))
    
    current = []
    for job in order:
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(current) + 1):
            test_perm = current[:pos] + [job] + current[pos:]
            ms = _makespan(pt, test_perm)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        current = current[:best_pos] + [job] + current[best_pos:]
        
    return current

def _spt_initial(pt):
    """SPT (Shortest Processing Time) on first machine."""
    n_jobs, n_machines = pt.shape
    order = list(np.argsort(pt[:, 0]))
    
    current = []
    for job in order:
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(current) + 1):
            test_perm = current[:pos] + [job] + current[pos:]
            ms = _makespan(pt, test_perm)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        current = current[:best_pos] + [job] + current[best_pos:]
        
    return current

def _random_initial(n_jobs, seed):
    """Random initial solution."""
    rng = np.random.RandomState(seed)
    return rng.permutation(n_jobs).tolist()

def _local_search(pt, perm, deadline):
    """
    Hill climbing with insertion neighborhood.
    Tries to move each job to every other position.
    """
    n_jobs, n_machines = pt.shape
    current_perm = perm[:]
    current_ms = _makespan(pt, current_perm)
    
    improved = True
    while improved:
        improved = False
        if time.time() > deadline:
            break
            
        for i in range(n_jobs):
            if time.time() > deadline:
                break
            job = current_perm[i]
            # Try moving job from i to every other position
            for k in range(n_jobs):
                if k == i or k == i - 1:
                    continue
                new_ms, new_perm = _makespan_delta_insert(pt, current_perm, job, i, k)
                if new_ms < current_ms:
                    current_perm = new_perm
                    current_ms = new_ms
                    improved = True
                    break  # Restart the outer loop
            if improved:
                break
                
    return current_perm, current_ms

def _simulated_annealing(pt, perm, deadline, T0=100.0, cooling=0.99):
    """
    Simulated annealing with insertion neighborhood.
    """
    n_jobs, n_machines = pt.shape
    current_perm = perm[:]
    current_ms = _makespan(pt, current_perm)
    best_perm = current_perm[:]
    best_ms = current_ms
    
    T = T0
    rng = np.random.RandomState(123)
    
    while time.time() < deadline and T > 1e-5:
        # Pick a random move
        i = rng.randint(n_jobs)
        k = rng.randint(n_jobs)
        if k == i or k == i - 1:
            continue
            
        job = current_perm[i]
        new_ms, new_perm = _makespan_delta_insert(pt, current_perm, job, i, k)
        
        delta = new_ms - current_ms
        if delta < 0 or rng.random() < np.exp(-delta / T):
            current_perm = new_perm
            current_ms = new_ms
            if current_ms < best_ms:
                best_ms = current_ms
                best_perm = current_perm[:]
                
        T *= cooling
        
    return best_perm, best_ms

def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan using multi-start local search.
    """
    pt = np.asarray(processing_times)
    n_jobs, n_machines = pt.shape
    
    start_time = time.time()
    deadline = start_time + 140  # 10s margin before 150s limit
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    if n_machines == 0:
        return list(range(n_jobs))
    
    # Generate multiple initial solutions
    candidates = []
    candidates.append(_neh_initial(pt))
    candidates.append(_palmer_initial(pt))
    candidates.append(_lpt_initial(pt))
    candidates.append(_spt_initial(pt))
    
    # Add a few random starts
    for seed in range(5):
        candidates.append(_random_initial(n_jobs, seed))
    
    best_perm = candidates[0]
    best_ms = _makespan(pt, best_perm)
    
    # Refine each candidate with local search
    for cand in candidates:
        if time.time() > deadline:
            break
        
        # Allocate time for each candidate
        remaining = deadline - time.time()
        if remaining < 1.0:
            break
            
        # Use local search
        perm, ms = _local_search(pt, cand, deadline)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm[:]
            
    # If we have time left, try simulated annealing on the best solution
    if time.time() < deadline - 5.0:
        sa_deadline = deadline - 2.0
        perm, ms = _simulated_annealing(pt, best_perm, sa_deadline)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm[:]
            
    return best_perm

# EVOLVE-BLOCK-END
