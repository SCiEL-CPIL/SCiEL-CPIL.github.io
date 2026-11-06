import numpy as np
import time
import random

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 90s limit

    # 1. Generate APs efficiently using numpy
    ap_indices = []
    for start in range(1, n + 1):
        for step in range(1, n + 1):
            terms = list(range(start - 1, n, step))
            if len(terms) >= 2:
                ap_indices.append(np.array(terms, dtype=np.int32))
                
    num_aps = len(ap_indices)
    
    # 2. Precompute pos_to_aps as numpy arrays for faster iteration
    pos_to_aps = [np.array([], dtype=np.int32) for _ in range(n)]
    for ap_id, idx in enumerate(ap_indices):
        for i in idx:
            pos_to_aps[i] = np.append(pos_to_aps[i], ap_id)
    # Convert to list of lists for faster Python iteration
    pos_to_aps = [arr.tolist() for arr in pos_to_aps]
    
    # 3. Helper functions
    def compute_initial_sums(signs):
        sums = np.zeros(num_aps, dtype=np.int32)
        for ap_id, idx in enumerate(ap_indices):
            sums[ap_id] = np.sum(signs[idx])
        return sums

    def local_search(signs, max_iters, deadline, rng):
        signs = signs.copy()
        ap_sums = compute_initial_sums(signs)
        
        # Histogram for O(1) max query - track max dynamically
        max_possible_sum = n
        hist = np.zeros(max_possible_sum + 1, dtype=np.int32)
        for s in ap_sums:
            hist[abs(s)] += 1
            
        # Track current max to avoid scanning histogram every time
        current_max = int(np.max(np.where(hist > 0, np.arange(len(hist)), 0)))
        best_signs = signs.copy()
        best_max = current_max
        
        temperature = 3.0
        cooling = 0.9995
        min_temp = 0.05
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
                
            i = rng.randint(0, n - 1)
            old_sign = signs[i]
            new_sign = -old_sign
            delta = 2 * new_sign
            
            affected = pos_to_aps[i]
            
            # Track max changes during this move
            old_max = current_max
            new_max = current_max
            
            # Apply change
            for ap_id in affected:
                old_abs = abs(ap_sums[ap_id])
                hist[old_abs] -= 1
                if old_abs == current_max and hist[old_abs] == 0:
                    # Need to find new max
                    pass
                ap_sums[ap_id] += delta
                new_abs = abs(ap_sums[ap_id])
                hist[new_abs] += 1
                if new_abs > new_max:
                    new_max = new_abs
            
            # If we lost the old max, scan down
            if hist[old_max] == 0:
                for k in range(old_max - 1, -1, -1):
                    if hist[k] > 0:
                        new_max = k
                        break
                else:
                    new_max = 0
            
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
                current_max = old_max
                    
            temperature *= cooling
            
        return best_signs, best_max

    # 4. Main Loop with Restarts
    best_overall_signs = None
    best_overall_max = n + 1
    
    # Start with Thue-Morse (often a good starting point for discrepancy problems)
    thue_morse = np.array([1 if bin(i).count('1') % 2 == 0 else -1 for i in range(n)], dtype=np.int8)
    best_signs, best_max = local_search(thue_morse, 8000, deadline, random.Random(42))
    best_overall_signs = best_signs
    best_overall_max = best_max
    
    # Random restarts
    seeds = [1, 2, 3, 4, 5, 10, 100, 1000, 2024, 3141, 2718, 9999, 12345, 54321, 777, 888, 999, 111, 222, 333, 444, 555, 666, 777, 888, 999, 1111, 2222, 3333, 4444]
    for seed in seeds:
        if time.time() > deadline - 5.0:
            break
        rng = random.Random(seed)
        signs = np.array([rng.choice([-1, 1]) for _ in range(n)], dtype=np.int8)
        best_signs, best_max = local_search(signs, 8000, deadline, rng)
        
        if best_max < best_overall_max:
            best_overall_max = best_max
            best_overall_signs = best_signs.copy()

    # Final polish: Greedy descent on the best solution found
    if best_overall_signs is not None:
        signs = best_overall_signs.copy()
        ap_sums = compute_initial_sums(signs)
        max_possible_sum = n
        hist = np.zeros(max_possible_sum + 1, dtype=np.int32)
        for s in ap_sums:
            hist[abs(s)] += 1
            
        current_max = int(np.max(np.where(hist > 0, np.arange(len(hist)), 0)))
        
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
                old_max = current_max
                new_max = current_max
                
                # Apply
                for ap_id in affected:
                    old_abs = abs(ap_sums[ap_id])
                    hist[old_abs] -= 1
                    ap_sums[ap_id] += delta
                    new_abs = abs(ap_sums[ap_id])
                    hist[new_abs] += 1
                    if new_abs > new_max:
                        new_max = new_abs
                
                # If we lost the old max, scan down
                if hist[old_max] == 0:
                    for k in range(old_max - 1, -1, -1):
                        if hist[k] > 0:
                            new_max = k
                            break
                    else:
                        new_max = 0
                
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
