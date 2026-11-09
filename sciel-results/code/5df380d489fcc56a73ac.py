import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 145.0
    
    N = 51975
    
    # Generate divisors
    divisors = []
    for d in range(2, N + 1):
        if N % d == 0:
            divisors.append(d)
    divisors.sort()
    
    # Precompute indices
    idx_cache = {}
    for m in divisors:
        for r in range(m):
            idx_cache[(m, r)] = np.arange(r, N, m)
            
    # State
    residues = {m: random.randint(0, m-1) for m in divisors}
    counts = np.zeros(N, dtype=np.int32)
    is_uncovered = np.ones(N, dtype=bool)
    uncovered_count = N
    
    def update_counts(m, r, sign):
        nonlocal uncovered_count
        idx = idx_cache[(m, r)]
        if sign == 1: # Add
            # Count how many were uncovered
            dec = np.sum(is_uncovered[idx])
            counts[idx] += 1
            is_uncovered[idx] = (counts[idx] == 0)
            uncovered_count -= dec
        else: # Remove
            # Count how many become uncovered
            inc = np.sum(counts[idx] == 1)
            counts[idx] -= 1
            is_uncovered[idx] = (counts[idx] == 0)
            uncovered_count += inc

    # Initialize
    for m in divisors:
        r = residues[m]
        update_counts(m, r, 1)
        
    best_uncovered = uncovered_count
    best_residues = residues.copy()
    
    def local_search():
        nonlocal uncovered_count, residues, best_uncovered, best_residues
        improved = True
        while improved:
            improved = False
            # Shuffle order to avoid bias
            random.shuffle(divisors)
            for m in divisors:
                r_old = residues[m]
                # Remove old
                update_counts(m, r_old, -1)
                
                best_r = r_old
                best_unc = uncovered_count
                
                # Try all r
                for r in range(m):
                    if r == r_old:
                        continue
                    idx = idx_cache[(m, r)]
                    # If we add r, how many new points covered?
                    # Points covered are those where is_uncovered is True
                    new_covered = np.sum(is_uncovered[idx])
                    new_unc = uncovered_count - new_covered
                    if new_unc < best_unc:
                        best_unc = new_unc
                        best_r = r
                
                # Apply best
                if best_r != r_old:
                    update_counts(m, best_r, 1)
                    residues[m] = best_r
                    uncovered_count = best_unc
                    improved = True
                else:
                    # Re-add old if we didn't change? 
                    # Wait, if best_r == r_old, we should re-add r_old
                    update_counts(m, r_old, 1)
                    # uncovered_count is already correct?
                    # update_counts(m, r_old, -1) reduced it.
                    # update_counts(m, r_old, 1) increases it back.
                    # So yes.
                
                if uncovered_count < best_uncovered:
                    best_uncovered = uncovered_count
                    best_residues = residues.copy()
                    
    # Iterated Local Search
    max_iter = 1000
    for _ in range(max_iter):
        if time.time() > deadline - 5:
            break
            
        local_search()
        
        if best_uncovered == 0:
            break
            
        # Perturbation
        # Change 1-3 random residues
        n_perturb = random.randint(1, 3)
        for _ in range(n_perturb):
            m = random.choice(divisors)
            old_r = residues[m]
            new_r = random.randint(0, m-1)
            if new_r != old_r:
                update_counts(m, old_r, -1)
                update_counts(m, new_r, 1)
                residues[m] = new_r

    # Restore best
    counts = np.zeros(N, dtype=np.int32)
    is_uncovered = np.ones(N, dtype=bool)
    uncovered_count = N
    for m in divisors:
        r = best_residues[m]
        update_counts(m, r, 1)
        
    return [[m, best_residues[m]] for m in divisors]
