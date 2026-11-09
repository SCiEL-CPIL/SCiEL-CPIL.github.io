"""Improved program for flow-shop permutation construction using NEH and local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _makespan(order, P):
    """
    Compute the makespan for a given job order.
    order: list or array of job indices
    P: np.ndarray of shape (n_jobs, n_machines)
    """
    if len(order) == 0:
        return 0.0
    
    n_machines = P.shape[1]
    order_arr = np.asarray(order, dtype=np.intp)
    P_sub = P[order_arr]  # shape (n_jobs, n_machines)
    
    # Vectorized flow-shop makespan computation
    # C[j, m] = max(C[j-1, m], C[j, m-1]) + P[j, m]
    # We can compute this iteratively over jobs, vectorizing over machines
    comp = np.zeros(n_machines)  # completion times of previous job on each machine
    for j in range(len(order_arr)):
        p_row = P_sub[j]
        # For each machine m: C[j,m] = max(comp[m], C[j,m-1]) + P[j,m]
        # This is a sequential dependency over m, so we can't fully vectorize
        # But we can use a trick: C[j,m] = cummax approach doesn't work directly
        # Let's just do the loop but in a tight manner
        prev = 0.0
        for m in range(n_machines):
            prev = max(comp[m], prev) + p_row[m]
            comp[m] = prev
    return comp[-1]

def _neh_heuristic(P):
    """
    Implement the NEH heuristic with incremental makespan evaluation.
    """
    n_jobs, n_machines = P.shape
    
    # 1. Calculate total processing times
    totals = P.sum(axis=1)
    
    # 2. Sort jobs by decreasing total processing time
    sorted_jobs = np.argsort(-totals)
    
    # 3. Construct permutation
    perm = []
    for j in sorted_jobs:
        best_pos = 0
        best_comp = float('inf')
        
        # Precompute completion times for the current perm prefix
        # For each position pos, we need C[j, M] when j is inserted at pos
        # We can compute the completion state after each prefix of perm
        
        # Build prefix completion states
        # prefix_comp[k] = completion times on each machine after processing first k jobs of perm
        prefix_comps = [np.zeros(n_machines)]  # after 0 jobs
        for idx in range(len(perm)):
            comp = prefix_comps[idx].copy()
            job_idx = perm[idx]
            prev = 0.0
            for m in range(n_machines):
                prev = max(comp[m], prev) + P[job_idx, m]
                comp[m] = prev
            prefix_comps.append(comp)
        
        # Now try inserting j at each position
        for pos in range(len(perm) + 1):
            # Get completion state before position pos
            comp_before = prefix_comps[pos].copy()
            
            # Process job j from state comp_before
            prev = 0.0
            for m in range(n_machines):
                prev = max(comp_before[m], prev) + P[j, m]
                comp_before[m] = prev
            
            # comp_before[-1] is C[j, M]
            if comp_before[-1] < best_comp:
                best_comp = comp_before[-1]
                best_pos = pos
        
        perm = perm[:best_pos] + [j] + perm[best_pos:]
        
    return perm

def _adjacent_swap_local_search(perm, P, deadline):
    """
    Perform adjacent-swap local search with time guard.
    """
    n_jobs = len(perm)
    if n_jobs < 2:
        return perm
    
    best_perm = list(perm)
    best_ms = _makespan(best_perm, P)
    
    improved = True
    while improved:
        improved = False
        for i in range(n_jobs - 1):
            if time.time() > deadline:
                break
            # Try swapping adjacent jobs at i and i+1
            new_perm = best_perm.copy()
            new_perm[i], new_perm[i+1] = new_perm[i+1], new_perm[i]
            
            ms = _makespan(new_perm, P)
            if ms < best_ms - 1e-9:
                best_ms = ms
                best_perm = new_perm
                improved = True
        
        if time.time() > deadline:
            break
    
    return best_perm

def order_jobs(processing_times):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit
    
    P = np.asarray(processing_times, dtype=float)
    n_jobs, n_machines = P.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # Step 1: NEH Heuristic
    perm = _neh_heuristic(P)
    best_perm = list(perm)
    best_ms = _makespan(best_perm, P)
    
    # Step 2: Adjacent-swap local search with time guard
    new_perm = _adjacent_swap_local_search(best_perm, P, deadline)
    new_ms = _makespan(new_perm, P)
    if new_ms < best_ms:
        best_perm = new_perm
        best_ms = new_ms
    
    # Step 3: If time remains, do another round of local search
    if time.time() < deadline - 2.0:
        new_perm = _adjacent_swap_local_search(best_perm, P, deadline)
        new_ms = _makespan(new_perm, P)
        if new_ms < best_ms:
            best_perm = new_perm
            best_ms = new_ms
    
    return list(best_perm)
# EVOLVE-BLOCK-END
