import numpy as np
import time
import random

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    # 1. Generate APs: Only APs with length >= 2 (singletons are trivial, sum = +/-1)
    ap_indices = []
    for start in range(1, n + 1):
        for step in range(1, n + 1):
            # Generate terms
            terms = list(range(start, n + 1, step))
            # Only include if length >= 2
            if len(terms) >= 2:
                idx = [t - 1 for t in terms]
                ap_indices.append(idx)
                
    num_aps = len(ap_indices)
    
    # 2. Precompute pos_to_aps
    pos_to_aps = [[] for _ in range(n)]
    for ap_id, idx in enumerate(ap_indices):
        for i in idx:
            pos_to_aps[i].append(ap_id)
            
    # Convert to arrays for faster iteration if needed, or keep lists
    # Lists are fine for ~1000 items total.
    
    # 3. Helper functions
    def compute_initial_sums(signs):
        sums = np.zeros(num_aps, dtype=np.int32)
        for ap_id, idx in enumerate(ap_indices):
            sums[ap_id] = np.sum(signs[idx])
        return sums
        
    def get_max_abs(sums):
        return int(np.max(np.abs(sums)))

    def local_search(signs, max_iters, deadline, rng):
        signs = signs.copy()
        ap_sums = compute_initial_sums(signs)
        current_max = get_max_abs(ap_sums)
        best_signs = signs.copy()
        best_max = current_max
        
        # Histogram for O(1) max? 
        # num_aps is ~1300. np.max is C-optimized and very fast.
        # Updating histogram: 
        # We need to update ap_sums and then find max.
        # If we use np.max, we scan 1300 elements.
        # If we use histogram, we scan 257 elements (max possible sum).
        # Histogram update cost: O(degree of node).
        # Let's use histogram for robustness.
        
        max_possible_sum = n
        hist = np.zeros(max_possible_sum + 1, dtype=np.int32)
        for s in ap_sums:
            hist[abs(s)] += 1
            
        def get_max_from_hist():
            for k in range(max_possible_sum, -1, -1):
                if hist[k] > 0:
                    return k
            return 0
            
        current_max = get_max_from_hist()
        
        temperature = 2.0
        cooling = 0.999
        min_temp = 0.01
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
                
            i = rng.randint(0, n - 1)
            old_sign = signs[i]
            new_sign = -old_sign
            delta = 2 * new_sign # -2 * old_sign
            
            # Update sums
            affected = pos_to_aps[i]
            
            # Apply change
            for ap_id in affected:
                old_abs = abs(ap_sums[ap_id])
                hist[old_abs] -= 1
                ap_sums[ap_id] += delta
                new_abs = abs(ap_sums[ap_id])
                hist[new_abs] += 1
                
            new_max = get_max_from_hist()
            delta_max = new_max - current_max
            
            # Acceptance criteria
            if delta_max <= 0 or (temperature > min_temp and rng.random() < np.exp(-delta_max / temperature)):
                signs[i] = new_sign
                current_max = new_max
                if current_max < best_max:
                    best_max = current_max
                    best_signs = signs.copy()
            else:
                # Revert
                signs[i] = old_sign
                for ap_id in affected:
                    old_abs = abs(ap_sums[ap_id])
                    hist[old_abs] -= 1
                    ap_sums[ap_id] -= delta
                    new_abs = abs(ap_sums[ap_id])
                    hist[new_abs] += 1
                    
            temperature *= cooling
            
        return best_signs, best_max

    # 4. Main Loop with Restarts
    best_overall_signs = None
    best_overall_max = n + 1
    
    # Start with Thue-Morse
    thue_morse = np.array([1 if bin(i).count('1') % 2 == 0 else -1 for i in range(n)], dtype=np.int8)
    best_signs, best_max = local_search(thue_morse, 15000, deadline, random.Random(42))
    best_overall_signs = best_signs
    best_overall_max = best_max
    
    # Random restarts
    seeds = [1, 2, 3, 4, 5, 10, 100, 1000, 2024, 3141, 2718, 9999, 12345, 54321]
    for seed in seeds:
        if time.time() > deadline - 5.0:
            break
        rng = random.Random(seed)
        signs = np.array([rng.choice([-1, 1]) for _ in range(n)], dtype=np.int8)
        best_signs, best_max = local_search(signs, 15000, deadline, rng)
        
        if best_max < best_overall_max:
            best_overall_max = best_max
            best_overall_signs = best_signs.copy()
            
        if best_overall_max <= 2:
            break
            
    # Final polish: Greedy descent on the best solution found
    if best_overall_signs is not None:
        signs = best_overall_signs.copy()
        ap_sums = compute_initial_sums(signs)
        max_possible_sum = n
        hist = np.zeros(max_possible_sum + 1, dtype=np.int32)
        for s in ap_sums:
            hist[abs(s)] += 1
            
        def get_max_from_hist():
            for k in range(max_possible_sum, -1, -1):
                if hist[k] > 0:
                    return k
            return 0
            
        current_max = get_max_from_hist()
        
        improved = True
        while improved and time.time() < deadline - 1.0:
            improved = False
            for i in range(n):
                if time.time() > deadline - 1.0:
                    break
                old_sign = signs[i]
                new_sign = -old_sign
                delta = 2 * new_sign
                
                affected = pos_to_aps[i]
                
                # Apply
                for ap_id in affected:
                    old_abs = abs(ap_sums[ap_id])
                    hist[old_abs] -= 1
                    ap_sums[ap_id] += delta
                    new_abs = abs(ap_sums[ap_id])
                    hist[new_abs] += 1
                    
                new_max = get_max_from_hist()
                
                if new_max < current_max:
                    current_max = new_max
                    signs[i] = new_sign
                    improved = True
                else:
                    # Revert
                    signs[i] = old_sign
                    for ap_id in affected:
                        old_abs = abs(ap_sums[ap_id])
                        hist[old_abs] -= 1
                        ap_sums[ap_id] -= delta
                        new_abs = abs(ap_sums[ap_id])
                        hist[new_abs] += 1
                        
        if current_max < best_overall_max:
            best_overall_max = current_max
            best_overall_signs = signs.copy()

    if best_overall_signs is None:
        rng = random.Random(42)
        return [rng.choice([-1, 1]) for _ in range(n)]
        
    return [int(s) for s in best_overall_signs]
