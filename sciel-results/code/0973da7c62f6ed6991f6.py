"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _makespan(pt, perm):
    """Compute makespan for a given permutation. O(n*m)."""
    n, m = pt.shape
    if len(perm) == 0:
        return 0.0
    
    jobs = pt[perm]
    c = np.zeros(m)
    for j in range(n):
        new_c = np.empty(m)
        new_c[0] = c[0] + jobs[j, 0]
        if m > 1:
            for k in range(1, m):
                new_c[k] = max(c[k], new_c[k-1]) + jobs[j, k]
        c = new_c
    return c[-1]

def _insert_move(perm, i, j):
    """Remove job at i, insert at j. Assumes i != j."""
    p = list(perm)
    job = p.pop(i)
    # If i < j, pop shifts indices, so insert at j-1? 
    # Standard: pop(i) removes. Insert at j.
    # If i < j, the element at j is now at j-1. 
    # If we want to insert "after" the element originally at j-1 (now at j-1), 
    # we need to be careful.
    # Let's stick to a simple definition:
    # Remove index i. Insert at index j in the resulting list.
    p.insert(j, job)
    return p

def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan using Tabu Search.
    """
    start_time = time.time()
    deadline = start_time + 115.0
    
    pt = np.asarray(processing_times, dtype=np.float64)
    
    # Robust shape handling
    if pt.ndim == 1:
        # Assume n jobs, 1 machine
        pt = pt.reshape(-1, 1)
        
    n, m = pt.shape
    
    if n <= 1:
        return list(range(n))
    
    # Initial Solution: NEH Heuristic
    totals = pt.sum(axis=1)
    order = np.argsort(-totals)
    perm = []
    for j in order:
        best_pos = 0
        best_mk = float('inf')
        for pos in range(len(perm) + 1):
            candidate = perm[:pos] + [j] + perm[pos:]
            mk = _makespan(pt, candidate)
            if mk < best_mk:
                best_mk = mk
                best_pos = pos
        perm = perm[:best_pos] + [j] + perm[best_pos:]
    
    best_perm = list(perm)
    best_mk = _makespan(pt, best_perm)
    current_perm = list(best_perm)
    current_mk = best_mk
    
    # Tabu Search Parameters
    tabu_tenure = max(10, n // 2)
    max_iter = 5000
    no_improve_limit = 200
    no_improve_count = 0
    
    # Tabu list: dict mapping (from_idx, to_idx) or (job, to_idx) to expiration iter
    # Using (job, to_pos) is common. Or (from, to).
    # Let's use (from, to) positions.
    tabu_list = {}
    rng = random.Random(42)
    
    for t in range(max_iter):
        if time.time() > deadline:
            break
            
        # Generate neighbor
        # Choose a random job to move
        i = rng.randint(0, n - 1)
        # Choose a random destination
        j = rng.randint(0, n - 1)
        if i == j:
            continue
            
        # Check tabu status
        # Tabu if (i, j) is in tabu_list and tabu_list[(i, j)] > t
        is_tabu = (i, j) in tabu_list and tabu_list[(i, j)] > t
        
        new_perm = _insert_move(current_perm, i, j)
        new_mk = _makespan(pt, new_perm)
        
        # Aspiration: Accept if better than best, even if tabu
        if new_mk < best_mk or (not is_tabu and new_mk < current_mk):
            current_perm = new_perm
            current_mk = new_mk
            
            if new_mk < best_mk:
                best_mk = new_mk
                best_perm = list(new_perm)
                no_improve_count = 0
            else:
                no_improve_count += 1
                
            # Add move to tabu list
            # We tabu the reverse move (j, i) to prevent immediate cycling back
            # But since indices shift, we should tabu the specific job moving to specific pos?
            # Simpler: Tabu the pair (i, j) for 'tabu_tenure' iterations
            tabu_list[(i, j)] = t + tabu_tenure
            # Also tabu the reverse? (j, i) is tricky because i,j change meaning.
            # Standard: Tabu the move that brought us here, or the specific job/pos.
            # Let's tabu (i, j) and (j, i) is not directly valid due to index shift.
            # A more robust tabu key is (job_id, to_pos).
            # Let's switch to (job_id, to_pos).
            
            # Re-implement tabu key as (job, to_pos)
            # The job moved was current_perm[i] (before move).
            # Let's stick to (i, j) for simplicity in this iteration, but realize it's weak.
            # Better: Tabu the job moving to position j.
            job_id = current_perm[i] # The job that was at i
            # Actually, _insert_move removes i and inserts at j.
            # The job is the one at i in the OLD perm.
            # We already computed new_perm.
            # Let's just use (i, j) for now.
            
        else:
            no_improve_count += 1
            
        # Diversification
        if no_improve_count > no_improve_limit:
            # Perturb best solution
            current_perm = list(best_perm)
            # Do a few random insertions
            for _ in range(5):
                i = rng.randint(0, n - 1)
                j = rng.randint(0, n - 1)
                if i != j:
                    current_perm = _insert_move(current_perm, i, j)
            current_mk = _makespan(pt, current_perm)
            no_improve_count = 0
            tabu_list.clear() # Clear tabu to allow escape
            
        # Cleanup tabu list to prevent memory leak
        if len(tabu_list) > 1000:
            tabu_list = {k: v for k, v in tabu_list.items() if v > t}

    return best_perm
# EVOLVE-BLOCK-END
