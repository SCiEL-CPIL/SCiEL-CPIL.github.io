"""Construct signs to minimize max absolute sum on any arithmetic progression in [1,256]."""
import math
import random
import time
import numpy as np

def construct():
    n = 256
    
    # Precompute all maximal arithmetic progressions as index arrays.
    # For a given start and step, the maximal AP covers all shorter APs with same start/step
    # via cumulative sums.
    ap_indices = []
    for step in range(1, n + 1):
        for start in range(step):
            idx = list(range(start, n, step))
            if idx:
                ap_indices.append(np.array(idx, dtype=np.int32))
    
    # Optimized discrepancy calculation
    def compute_max_discrepancy(s, threshold=None):
        max_abs = 1 # Singletons guarantee max >= 1
        for idx in ap_indices:
            subseq = s[idx]
            csum = np.cumsum(subseq)
            local_max = np.max(np.abs(csum))
            
            if local_max > max_abs:
                max_abs = local_max
                if threshold is not None and max_abs >= threshold:
                    return max_abs
        return max_abs

    def is_prime(x):
        if x < 2:
            return False
        if x % 2 == 0:
            return x == 2
        if x % 3 == 0:
            return x == 3
        i = 5
        while i * i <= x:
            if x % i == 0 or x % (i + 2) == 0:
                return False
            i += 6
        return True

    # Step 1: Generate multiple initial candidates based on Legendre symbols
    candidates = []
    # Find primes near n
    primes = []
    for p in range(max(2, n - 50), n + 50):
        if is_prime(p):
            primes.append(p)
            
    for p in primes[:10]:
        legendre = []
        for a in range(1, n + 1):
            if a % p == 0:
                legendre.append(1)
            else:
                val = pow(a, (p - 1) // 2, p)
                if val == 1:
                    legendre.append(1)
                else:
                    legendre.append(-1)
        candidates.append(np.array(legendre, dtype=np.int8))
        
    # Add a random candidate
    rng = random.Random(42)
    candidates.append(np.array([rng.choice([-1, 1]) for _ in range(n)], dtype=np.int8))

    # Evaluate all candidates and pick the best
    best_signs = None
    best_max = float('inf')
    
    for cand in candidates:
        curr_max = compute_max_discrepancy(cand)
        if curr_max < best_max:
            best_max = curr_max
            best_signs = cand.copy()

    signs = best_signs.copy()
    current_max = best_max

    # Step 2: Iterated Local Search with Targeted Flips
    start_time = time.time()
    time_limit = 55.0
    
    def local_search(s, current_max):
        improved = True
        # Limit iterations to avoid infinite loops if no improvement is found
        max_iter = 100
        iter_count = 0
        
        while improved and iter_count < max_iter:
            improved = False
            iter_count += 1
            
            # 1. Find the APs with the highest discrepancy
            max_val = 0
            worst_aps = []
            
            for idx in ap_indices:
                subseq = s[idx]
                csum = np.cumsum(subseq)
                local_max = np.max(np.abs(csum))
                
                if local_max > max_val:
                    max_val = local_max
                    worst_aps = [idx]
                elif local_max == max_val:
                    worst_aps.append(idx)
            
            if max_val < current_max:
                current_max = max_val
            
            # 2. Try to flip signs in the worst APs to reduce the max
            # Collect unique indices from the worst APs
            candidate_indices = set()
            for idx in worst_aps:
                candidate_indices.update(idx.tolist())
            
            # Limit the number of flip attempts per iteration to save time
            cand_list = list(candidate_indices)
            rng.shuffle(cand_list)
            
            flip_found = False
            # Try up to 50 flips in the worst APs
            for i in cand_list[:50]:
                old_val = s[i]
                s[i] = -old_val
                
                # Check if this flip improves the global max
                new_max = compute_max_discrepancy(s, current_max)
                
                if new_max < current_max:
                    current_max = new_max
                    improved = True
                    flip_found = True
                    break
                else:
                    s[i] = old_val
            
            if not flip_found:
                # No improvement found in the current set of worst APs
                # We could expand to next worst APs, but for now, break to allow perturbation
                break
        
        return s, current_max

    # Main ILS loop
    no_improve_count = 0
    max_no_improve = 5
    
    while time.time() - start_time < time_limit:
        signs, current_max = local_search(signs, current_max)
        
        if current_max < best_max:
            best_max = current_max
            best_signs = signs.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            
        if no_improve_count >= max_no_improve:
            # Perturb: Flip more random signs to escape local minimum
            trial = best_signs.copy()
            num_flips = rng.randint(10, 20)
            flip_indices = rng.sample(range(n), num_flips)
            for i in flip_indices:
                trial[i] = -trial[i]
            
            signs = trial
            current_max = compute_max_discrepancy(signs)
            no_improve_count = 0

    return [int(x) for x in best_signs]
