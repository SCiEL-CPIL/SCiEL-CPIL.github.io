import time
import random
import math
import numpy as np
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    n = 20
    max_val = 10000
    
    # 1. Generate pool of 7-smooth numbers (2^a * 3^b * 5^c * 7^d) <= 10000
    # This pool is dense enough to allow flexible optimization but structured enough
    # that products and sums tend to stay within the pool or close to it, 
    # potentially reducing the size of A+A and A*A compared to arbitrary integers.
    smooth_pool = set()
    for a in range(0, 14):  # 2^13 = 8192
        v2 = 1 << a
        if v2 > max_val: break
        for b in range(0, 9): # 3^8 = 6561
            v3 = v2 * (3 ** b)
            if v3 > max_val: break
            for c in range(0, 6): # 5^5 = 3125
                v5 = v3 * (5 ** c)
                if v5 > max_val: break
                for d in range(0, 5): # 7^4 = 2401
                    v7 = v5 * (7 ** d)
                    if v7 > max_val: break
                    smooth_pool.add(v7)
    
    smooth_list = sorted(list(smooth_pool))
    num_smooth = len(smooth_list)
    
    # Precompute all pairs (i, j) with i <= j for n=20
    # There are 20*21/2 = 210 pairs
    pairs_i = []
    pairs_j = []
    for i in range(n):
        for j in range(i, n):
            pairs_i.append(i)
            pairs_j.append(j)
            
    def evaluate_fast(A):
        """
        Compute max(|A+A|, |A*A|) for set A using pure Python sets.
        This is faster than numpy for small n due to lower overhead.
        """
        sums = set()
        prods = set()
        for i in range(n):
            ai = A[i]
            for j in range(i, n):
                aj = A[j]
                sums.add(ai + aj)
                prods.add(ai * aj)
        return max(len(sums), len(prods))
        
    def evaluate_numpy(A):
        """Fallback or verification, but we stick to fast for search."""
        A_arr = np.array(A, dtype=np.int64)
        sums = A_arr[pairs_i] + A_arr[pairs_j]
        prods = A_arr[pairs_i] * A_arr[pairs_j]
        sumset_size = len(np.unique(sums))
        prodset_size = len(np.unique(prods))
        return max(sumset_size, prodset_size)
        
    # 2. Multi-Start Simulated Annealing
    best_score = float('inf')
    best_set = None
    
    # We will run multiple restarts. 
    # Each restart:
    # 1. Initialize with a random subset of smooth numbers (or a structured one)
    # 2. Run SA to minimize the score
    
    # To improve initial quality, we can use a greedy "compact" start occasionally,
    # but random starts help escape local minima.
    
    num_restarts = 1000 # Limit by time instead
    restart_count = 0
    
    while time.time() < deadline - 5:
        restart_count += 1
        
        # Initialization Strategy:
        # Mix of random and structured (small numbers)
        if restart_count % 3 == 0:
            # Structured: Start with smallest smooth numbers
            # This tends to have small sums and products
            init_indices = list(range(min(num_smooth, n * 2)))
            random.shuffle(init_indices)
            current_indices = init_indices[:n]
        else:
            # Random
            current_indices = random.sample(range(num_smooth), n)
            
        current_vals = [smooth_list[i] for i in current_indices]
        current_score = evaluate_fast(current_vals)
        
        if current_score < best_score:
            best_score = current_score
            best_set = current_vals[:]
            
        # Simulated Annealing Parameters
        T = 100.0 # Initial temperature (score differences are usually small integers)
        cooling_rate = 0.995
        min_T = 0.1
        
        # SA Loop
        steps_this_restart = 0
        max_steps_per_restart = 50000 # Cap to ensure we do multiple restarts
        
        while T > min_T and steps_this_restart < max_steps_per_restart:
            if time.time() > deadline - 2:
                break
                
            # Move: Swap two elements
            i, j = random.sample(range(n), 2)
            
            # Pick a new value for index i from the pool
            # To make moves more local, we could pick neighbors in the sorted list,
            # but random sampling from the whole pool explores better.
            # However, for smooth numbers, "local" in value might be better?
            # Let's try picking a random index from the pool.
            new_idx = random.randint(0, num_smooth - 1)
            
            # If new_idx is already in the set (at another position), we are just swapping values?
            # No, we are replacing A[i] with smooth_list[new_idx].
            # We must ensure distinctness.
            if new_idx == current_indices[i]:
                continue
            if new_idx in current_indices:
                # If the new value is already present elsewhere, we can't add it directly 
                # unless we swap. But our move is "replace". 
                # If we replace A[i] with a value that exists at A[k], we lose distinctness.
                # So we must either skip or perform a swap.
                # Simplest: Skip if collision.
                continue
                
            old_idx = current_indices[i]
            old_val = current_vals[i]
            new_val = smooth_list[new_idx]
            
            # Apply move
            current_indices[i] = new_idx
            current_vals[i] = new_val
            
            new_score = evaluate_fast(current_vals)
            delta = new_score - current_score
            
            if delta < 0 or random.random() < math.exp(-delta / T):
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_set = current_vals[:]
            else:
                # Revert
                current_indices[i] = old_idx
                current_vals[i] = old_val
                
            T *= cooling_rate
            steps_this_restart += 1
            
        # Optional: Local refinement after SA
        # A simple hill climbing pass on the exact score
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for i in range(n):
                old_idx = current_indices[i]
                old_val = current_vals[i]
                # Try swapping with a few random candidates
                for _ in range(10):
                    new_idx = random.randint(0, num_smooth - 1)
                    if new_idx == old_idx or new_idx in current_indices:
                        continue
                    current_indices[i] = new_idx
                    current_vals[i] = smooth_list[new_idx]
                    trial_score = evaluate_fast(current_vals)
                    if trial_score < current_score:
                        current_score = trial_score
                        improved = True
                        if current_score < best_score:
                            best_score = current_score
                            best_set = current_vals[:]
                    else:
                        # Revert
                        current_indices[i] = old_idx
                        current_vals[i] = old_val

    return sorted(best_set)

