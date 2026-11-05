import numpy as np
import time
import random

def construct():
    N = 51975
    # Find divisors of 51975
    divisors = []
    for i in range(2, int(np.sqrt(N)) + 1):
        if N % i == 0:
            divisors.append(i)
            if i != N // i:
                divisors.append(N // i)
    divisors.append(N)
    divisors = sorted(list(set(divisors)))
    divisors = [d for d in divisors if d > 1]
    
    num_mods = len(divisors)
    
    # Precompute index arrays for each divisor to speed up coverage checks
    # idx_arrays[i] is the array of indices covered by modulus divisors[i] with residue 0?
    # No, we need to handle arbitrary residues.
    # We can precompute the "base" indices for residue 0, but shifting is hard with arange.
    # However, we can precompute the list of indices for each (m, r)?
    # Total memory: sum(m) * 8 bytes? No, indices are int32/int64.
    # Sum(m) ~ 1.16 million indices. 1.16M * 8 bytes = ~9MB. This is fine.
    # Precomputing all index arrays allows O(1) access to the slice.
    
    idx_cache = {}
    for i, m in enumerate(divisors):
        idx_cache[i] = {}
        # We might not want to precompute ALL residues for large m if memory is tight, 
        # but 1.16M indices is small.
        for r in range(m):
            idx_cache[i][r] = np.arange(r, N, m)
            
    # Simulated Annealing
    start_time = time.time()
    timeout = 115.0
    
    # Initial solution: Random
    residues = [random.randint(0, m - 1) for m in divisors]
    counts = np.zeros(N, dtype=np.int32)
    for i, r in enumerate(residues):
        counts[idx_cache[i][r]] += 1
        
    current_score = np.sum(counts == 0)
    best_score = current_score
    best_residues = residues.copy()
    
    T = 50.0
    alpha = 0.99
    T_min = 0.01
    
    # We will run multiple "restarts" or just one long SA?
    # One long SA is good.
    
    while time.time() - start_time < timeout and T > T_min:
        # Perform some steps at current temperature
        steps = 500
        for _ in range(steps):
            if time.time() - start_time > timeout:
                break
                
            # Pick random modulus
            i = random.randint(0, num_mods - 1)
            m = divisors[i]
            
            # Pick random new residue
            old_r = residues[i]
            new_r = random.randint(0, m - 1)
            if new_r == old_r:
                continue
                
            # Calculate delta
            idx_old = idx_cache[i][old_r]
            idx_new = idx_cache[i][new_r]
            
            # inc: elements in idx_old that were covered only by this class (count==1)
            inc = np.sum(counts[idx_old] == 1)
            # dec: elements in idx_new that were uncovered (count==0)
            dec = np.sum(counts[idx_new] == 0)
            
            delta = inc - dec
            
            # Metropolis criterion
            if delta < 0 or random.random() < np.exp(-delta / T):
                # Accept
                counts[idx_old] -= 1
                counts[idx_new] += 1
                residues[i] = new_r
                current_score += delta
                
                if current_score < best_score:
                    best_score = current_score
                    best_residues = residues.copy()
                    
        T *= alpha
        
    # If we hit 0, stop early?
    if best_score == 0:
        pass
        
    classes = [[divisors[i], int(best_residues[i])] for i in range(num_mods)]
    return classes
