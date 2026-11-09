"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _makespan(perm, pt):
    """Compute makespan for a given permutation."""
    if len(perm) == 0:
        return 0.0
    m = pt.shape[1]
    c = np.zeros(m, dtype=np.float64)
    for job_idx in perm:
        prev_c = 0.0
        for k in range(m):
            val = c[k]
            if val < prev_c:
                val = prev_c
            val += pt[job_idx, k]
            c[k] = val
            prev_c = val
    return c[-1]

def _makespan_fast(perm, pt):
    """Faster makespan using list for perm and local variables."""
    m = pt.shape[1]
    c = [0.0] * m
    for job_idx in perm:
        prev_c = 0.0
        row = pt[job_idx]
        for k in range(m):
            val = c[k]
            if val < prev_c:
                val = prev_c
            val += row[k]
            c[k] = val
            prev_c = val
    return c[-1]

def _neh(processing_times):
    """NEH heuristic to generate initial solution."""
    n, m = processing_times.shape
    if n <= 1:
        return list(range(n))
    
    totals = processing_times.sum(axis=1)
    order = np.argsort(-totals).tolist()
    
    perm = []
    for job in order:
        best_pos = 0
        best_ms = float('inf')
        for k in range(len(perm) + 1):
            trial_perm = perm[:k] + [job] + perm[k:]
            ms = _makespan_fast(trial_perm, processing_times)
            if ms < best_ms:
                best_ms = ms
                best_pos = k
        perm.insert(best_pos, job)
    
    return perm

def _insertion(perm, processing_times):
    """Local search using insertion moves."""
    n = len(perm)
    m = processing_times.shape[1]
    current_ms = _makespan_fast(perm, processing_times)
    
    improved = True
    while improved:
        improved = False
        for i in range(n):
            job = perm[i]
            for j in range(n):
                if i == j:
                    continue
                # Move job from i to j
                new_perm = perm.copy()
                new_perm.pop(i)
                # Adjust j if needed
                if i < j:
                    j -= 1
                new_perm.insert(j, job)
                
                new_ms = _makespan_fast(new_perm, processing_times)
                if new_ms < current_ms:
                    perm = new_perm
                    current_ms = new_ms
                    improved = True
                    break
            if improved:
                break
    
    return perm, current_ms

def order_jobs(processing_times):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n, m = processing_times.shape
    
    if n <= 1:
        return list(range(n))
    
    # 1. Generate initial solution using NEH
    best_perm = _neh(processing_times)
    best_ms = _makespan_fast(best_perm, processing_times)
    
    # 2. Polish with insertion local search
    best_perm, best_ms = _insertion(best_perm, processing_times)
    
    # 3. Simulated Annealing with multiple restarts
    random.seed(42)
    
    # Precompute some data for faster evaluation
    pt = processing_times
    
    # SA parameters
    T_init = best_ms * 0.1
    T_min = best_ms * 0.001
    alpha = 0.999
    
    # Multi-start SA
    n_restarts = 5
    time_per_restart = (deadline - time.time()) / max(1, n_restarts)
    
    for restart in range(n_restarts):
        if time.time() > deadline - 5.0:
            break
        
        # For restart 0, use the best found so far
        if restart == 0:
            current_perm = best_perm.copy()
            current_ms = best_ms
        else:
            # Perturb the best solution
            current_perm = best_perm.copy()
            # Apply several random swaps
            n_perturb = min(n, random.randint(3, 10))
            for _ in range(n_perturb):
                i, j = random.sample(range(n), 2)
                current_perm[i], current_perm[j] = current_perm[j], current_perm[i]
            current_ms = _makespan_fast(current_perm, pt)
        
        T = T_init
        stall_count = 0
        max_stall = 1000
        
        while time.time() < deadline - 5.0:
            if T < T_min:
                break
            
            # Generate neighbor: random swap
            i, j = random.sample(range(n), 2)
            new_perm = current_perm.copy()
            new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
            
            new_ms = _makespan_fast(new_perm, pt)
            delta = new_ms - current_ms
            
            if delta < 0 or random.random() < np.exp(-delta / T):
                current_perm = new_perm
                current_ms = new_ms
                stall_count = 0
                
                if current_ms < best_ms:
                    best_ms = current_ms
                    best_perm = current_perm.copy()
            else:
                stall_count += 1
            
            T *= alpha
            
            if stall_count > max_stall:
                break
        
        # Polish the best found in this restart with insertion
        if time.time() < deadline - 10.0:
            polished_perm, polished_ms = _insertion(best_perm.copy(), pt)
            if polished_ms < best_ms:
                best_ms = polished_ms
                best_perm = polished_perm
    
    # Final polish
    if time.time() < deadline - 5.0:
        polished_perm, polished_ms = _insertion(best_perm.copy(), pt)
        if polished_ms < best_ms:
            best_ms = polished_ms
            best_perm = polished_perm
    
    return list(best_perm)
# EVOLVE-BLOCK-END
