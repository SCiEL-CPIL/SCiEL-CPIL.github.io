import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 115.0
    
    N = 51975
    
    # Get divisors
    divisors = []
    for d in range(2, N + 1):
        if N % d == 0:
            divisors.append(d)
    divisors.sort()
    
    # State
    counts = np.zeros(N, dtype=np.int32)
    uncovered = N
    residues = {m: 0 for m in divisors} # m -> r
    
    def update_add(m, r):
        nonlocal uncovered
        idx = np.arange(r, N, m)
        dec = np.sum(counts[idx] == 0)
        counts[idx] += 1
        uncovered -= dec
        
    def update_remove(m, r):
        nonlocal uncovered
        idx = np.arange(r, N, m)
        inc = np.sum(counts[idx] == 1)
        counts[idx] -= 1
        uncovered += inc
    
    # Initialize counts with random residues
    for m in divisors:
        r = random.randint(0, m-1)
        residues[m] = r
        update_add(m, r)
        
    best_uncovered = uncovered
    best_residues = residues.copy()
    
    # Simulated Annealing approach
    # Temperature starts high, decreases over time
    T = 1.0
    T_min = 0.001
    cooling_rate = 0.9995
    
    while time.time() < deadline - 5 and T > T_min:
        # Shuffle order
        order = divisors.copy()
        random.shuffle(order)
        
        for m in order:
            if time.time() > deadline - 5:
                break
            if T <= T_min:
                break
                
            current_r = residues[m]
            best_r = current_r
            best_local_uncovered = uncovered
            
            # Remove current
            update_remove(m, current_r)
            
            # Try a random subset of residues (not all, to save time)
            # For small m, try all; for large m, try a sample
            if m <= 100:
                candidate_rs = list(range(m))
            else:
                sample_size = min(m, 50)
                candidate_rs = random.sample(range(m), sample_size)
            
            for r in candidate_rs:
                if r == current_r:
                    continue
                update_add(m, r)
                
                delta = uncovered - best_local_uncovered
                # Accept if better, or with probability exp(-delta/T) if worse
                if delta <= 0 or random.random() < np.exp(-delta / T):
                    best_local_uncovered = uncovered
                    best_r = r
                else:
                    update_remove(m, r)
                    continue
                update_remove(m, r)
            
            # Restore best
            if best_r != current_r:
                update_add(m, best_r)
                residues[m] = best_r
            else:
                update_add(m, current_r)
            
            # Update global best
            if uncovered < best_uncovered:
                best_uncovered = uncovered
                best_residues = residues.copy()
                
        # Cool down
        T *= cooling_rate
        
        # Occasionally do a larger perturbation if stuck
        if random.random() < 0.1:
            num_perturb = random.randint(2, 5)
            for _ in range(num_perturb):
                m = random.choice(divisors)
                new_r = random.randint(0, m-1)
                update_remove(m, residues[m])
                update_add(m, new_r)
                residues[m] = new_r
    
    # Final coordinate descent pass to polish
    for _ in range(10):
        if time.time() > deadline - 5:
            break
        improved = False
        order = divisors.copy()
        random.shuffle(order)
        
        for m in order:
            if time.time() > deadline - 5:
                break
                
            current_r = residues[m]
            best_r = current_r
            best_local_uncovered = uncovered
            
            update_remove(m, current_r)
            
            for r in range(m):
                if r == current_r:
                    continue
                update_add(m, r)
                if uncovered < best_local_uncovered:
                    best_local_uncovered = uncovered
                    best_r = r
                update_remove(m, r)
            
            if best_r != current_r:
                update_add(m, best_r)
                residues[m] = best_r
                improved = True
            else:
                update_add(m, current_r)
            
            if uncovered < best_uncovered:
                best_uncovered = uncovered
                best_residues = residues.copy()
                
        if not improved:
            break
            
    # Reconstruct best classes
    classes = []
    for m in divisors:
        classes.append([m, best_residues[m]])
        
    return classes
