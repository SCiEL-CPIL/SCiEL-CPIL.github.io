"""
Iterated Greedy (IG) with NEH construction and Or-opt local search.
This strategy fundamentally differs from Simulated Annealing by using a
destructive-constructive restart mechanism rather than stochastic temperature
acceptance. It aggressively destroys a portion of the current solution,
re-inserts jobs using a greedy NEH-like heuristic, and then applies local
search. This allows for larger structural changes to the permutation, helping
to escape the local minima that trap standard SA or simple hill-climbing.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _makespan(perm, pt):
    """
    Compute makespan for a given permutation.
    Uses a loop over jobs but vectorized operations where possible, though the
    dependency chain C[j,m] = max(C[j-1,m], C[j,m-1]) + p[j,m] makes full
    vectorization difficult without loops.
    """
    n_jobs = len(perm)
    if n_jobs == 0:
        return 0.0
    
    p = pt[perm]
    n_machines = p.shape[1]
    
    if n_jobs == 1:
        return np.sum(p[0])
        
    # C[j, m]
    C = np.zeros((n_jobs, n_machines))
    C[0] = p[0]
    
    for j in range(1, n_jobs):
        C[j, 0] = C[j-1, 0] + p[j, 0]
        for m in range(1, n_machines):
            C[j, m] = max(C[j-1, m], C[j, m-1]) + p[j, m]
            
    return C[-1, -1]

def _insert_job(perm, job, pt):
    """
    Insert a single job into the permutation at the position that minimizes makespan.
    """
    if not perm:
        return [job]
        
    best_cost = float('inf')
    best_pos = 0
    
    # Pre-compute p for the job to avoid repeated indexing
    p_job = pt[job]
    
    for pos in range(len(perm) + 1):
        candidate = perm[:pos] + [job] + perm[pos:]
        cost = _makespan(candidate, pt)
        if cost < best_cost:
            best_cost = cost
            best_pos = pos
            
    return perm[:best_pos] + [job] + perm[best_pos:]

def _neh_order(pt):
    """
    NEH heuristic: Sort jobs by total processing time descending, insert one by one.
    """
    n_jobs, _ = pt.shape
    totals = pt.sum(axis=1)
    sorted_jobs = list(np.argsort(-totals))
    
    perm = []
    for job in sorted_jobs:
        perm = _insert_job(perm, job, pt)
        
    return perm

def _or_opt_local_search(perm, pt, max_iterations=1000, time_limit=5.0):
    """
    Local search using Or-opt moves (moving segments of length 1, 2, 3).
    Returns the best permutation found.
    """
    current_perm = list(perm)
    current_cost = _makespan(current_perm, pt)
    best_perm = list(current_perm)
    best_cost = current_cost
    
    n_jobs = len(current_perm)
    start_time = time.time()
    iterations = 0
    no_improvement_count = 0
    
    while iterations < max_iterations and (time.time() - start_time) < time_limit:
        # Pick a random move
        seg_len = np.random.choice([1, 2, 3], p=[0.6, 0.3, 0.1])
        if n_jobs < seg_len:
            seg_len = 1
            
        start = np.random.randint(0, n_jobs - seg_len + 1)
        segment = current_perm[start:start+seg_len]
        
        # Remove segment
        remaining = current_perm[:start] + current_perm[start+seg_len:]
        
        # Try all insertion positions
        best_local_cost = current_cost
        best_local_pos = start # If no improvement, keep original
        
        for pos in range(len(remaining) + 1):
            if pos == start and len(remaining) >= start:
                # This corresponds to the original position roughly, but indices shifted.
                # We can skip if it results in same permutation, but checking is costly.
                # Just evaluate.
                pass
                
            candidate = remaining[:pos] + segment + remaining[pos:]
            cost = _makespan(candidate, pt)
            
            if cost < best_local_cost:
                best_local_cost = cost
                best_local_pos = pos
                best_local_perm = candidate
                
        if best_local_cost < current_cost:
            current_perm = best_local_perm
            current_cost = best_local_cost
            no_improvement_count = 0
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_perm = list(current_perm)
        else:
            no_improvement_count += 1
            # If stuck for a while, restart local search? Or just continue.
            # For IG, we want to finish LS quickly.
            if no_improvement_count > 50:
                break
                
        iterations += 1
        
    return best_perm, best_cost

def order_jobs(processing_times):
    """
    Main entry point: Iterated Greedy for flow-shop.
    """
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    start_total = time.time()
    deadline = 145.0 # Safety margin
    
    # 1. Initial solution via NEH
    current_perm = _neh_order(processing_times)
    current_cost = _makespan(current_perm, processing_times)
    
    best_perm = list(current_perm)
    best_cost = current_cost
    
    # 2. Initial Local Search
    current_perm, current_cost = _or_opt_local_search(current_perm, processing_times, time_limit=2.0)
    if current_cost < best_cost:
        best_cost = current_cost
        best_perm = list(current_perm)
    
    # 3. Iterated Greedy Loop
    # Parameters
    d_min = max(1, int(n_jobs * 0.1))
    d_max = max(1, int(n_jobs * 0.3))
    
    while time.time() < deadline:
        # Check time for the destruction/reconstruction phase
        if time.time() > deadline - 10.0:
            break
            
        # Destroy: Remove d random jobs
        d = np.random.randint(d_min, d_max + 1)
        if d >= n_jobs:
            d = n_jobs - 1
            
        # Select jobs to remove
        removed_jobs = []
        remaining_perm = list(current_perm)
        
        for _ in range(d):
            if not remaining_perm:
                break
            idx = np.random.randint(0, len(remaining_perm))
            job = remaining_perm.pop(idx)
            removed_jobs.append(job)
            
        # Reconstruct: Insert removed jobs one by one using greedy insertion
        # Order of insertion can matter. Let's try original order or random.
        # Greedy insertion usually works well.
        for job in removed_jobs:
            current_perm = _insert_job(current_perm, job, processing_times)
            
        # Local Search on the new solution
        current_perm, current_cost = _or_opt_local_search(current_perm, processing_times, time_limit=5.0)
        
        # Acceptance: Accept if better, or accept worse with some probability?
        # Standard IG often uses a simple acceptance or deterministic.
        # Let's use a simple probabilistic acceptance to allow exploration,
        # but biased towards better solutions.
        # Actually, simple "accept if better" is often strong for IG.
        # Let's try: accept if better, else accept with small probability to escape.
        if current_cost < best_cost:
            best_cost = current_cost
            best_perm = list(current_perm)
            
        # If we didn't improve, we still update current_perm to the result of LS
        # to allow further exploration from this new point.
        # However, if it's much worse, we might want to revert to best?
        # Standard IG updates current to the new solution regardless, relying on
        # the constructive heuristic to bring it back.
        
        # To be safe and effective, let's only keep the new current if it's not
        # catastrophically worse, or just always keep it.
        # Let's always keep it, as _insert_job is greedy.
        
        # Check time
        if time.time() > deadline - 5.0:
            break
            
    return best_perm
# EVOLVE-BLOCK-END
