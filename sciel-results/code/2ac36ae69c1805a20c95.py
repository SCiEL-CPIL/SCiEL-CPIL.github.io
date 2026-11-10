import numpy as np
import random
import time

def construct():
    # Generate primes <= 1000
    primes = [p for p in range(2, 1001) if all(p % q for q in range(2, int(p ** 0.5) + 1))]
    n_primes = len(primes)
    
    # Initial residues: random
    residues = [random.randint(0, p - 1) for p in primes]
    
    # State variables
    L_target = 100
    x_vals = np.arange(1, L_target + 1)
    depths = np.zeros(L_target, dtype=np.int32)
    
    # Function to compute depths from scratch
    def compute_depths(L, res):
        x = np.arange(1, L + 1)
        d = np.zeros(L, dtype=np.int32)
        for i, p in enumerate(primes):
            r = res[i]
            d += (x % p == r)
        return x, d

    x_vals, depths = compute_depths(L_target, residues)
    energy = int(np.sum(np.maximum(0, 2 - depths)))
    
    best_L = 0
    best_residues = residues.copy()
    
    start_time = time.time()
    time_limit = 110
    
    # SA parameters
    T = 10.0
    cooling = 0.9999
    min_T = 0.01
    
    while time.time() - start_time < time_limit:
        if energy == 0:
            if L_target > best_L:
                best_L = L_target
                best_residues = residues.copy()
            
            new_L = int(L_target * 1.05) + 1
            if new_L > 5000:
                new_L = 5000
                
            if new_L == L_target:
                T = max(T, 5.0)
                continue
                
            L_target = new_L
            x_vals, depths = compute_depths(L_target, residues)
            energy = int(np.sum(np.maximum(0, 2 - depths)))
            T = max(T, 5.0)
            continue
            
        moves = 500
        for _ in range(moves):
            if T < min_T:
                break
                
            i = random.randint(0, n_primes - 1)
            p = primes[i]
            old_r = residues[i]
            
            new_r = random.randint(0, p - 1)
            if new_r == old_r:
                continue
            
            delta_E = 0
            
            # Decrease depth for old_r
            k_min = (1 - old_r + p - 1) // p
            k_max = (L_target - old_r) // p
            
            for k in range(k_min, k_max + 1):
                x = k * p + old_r
                idx = x - 1
                old_depth = depths[idx]
                new_depth = old_depth - 1
                
                old_penalty = max(0, 2 - old_depth)
                new_penalty = max(0, 2 - new_depth)
                delta_E += new_penalty - old_penalty
                
                depths[idx] = new_depth
                
            # Increase depth for new_r
            k_min = (1 - new_r + p - 1) // p
            k_max = (L_target - new_r) // p
            
            for k in range(k_min, k_max + 1):
                x = k * p + new_r
                idx = x - 1
                old_depth = depths[idx]
                new_depth = old_depth + 1
                
                old_penalty = max(0, 2 - old_depth)
                new_penalty = max(0, 2 - new_depth)
                delta_E += new_penalty - old_penalty
                
                depths[idx] = new_depth
            
            if delta_E < 0 or random.random() < np.exp(-delta_E / T):
                residues[i] = new_r
                energy += delta_E
            else:
                # Revert
                for k in range(k_min, k_max + 1):
                    x = k * p + old_r
                    idx = x - 1
                    depths[idx] += 1
                for k in range(k_min, k_max + 1):
                    x = k * p + new_r
                    idx = x - 1
                    depths[idx] -= 1
            
            T *= cooling
            
        if T < min_T:
            T = 10.0

    return best_residues
