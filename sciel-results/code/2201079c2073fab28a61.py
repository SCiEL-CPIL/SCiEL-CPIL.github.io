"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import random

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 90s limit

    # Precompute all arithmetic progressions of length >= 2 within [1, 256]
    ap_indices = []
    for start in range(1, n + 1):
        for step in range(1, n + 1):
            terms = list(range(start, n + 1, step))
            if len(terms) >= 2:
                ap_indices.append([t - 1 for t in terms])
    
    num_aps = len(ap_indices)
    
    # Build position-to-AP mapping
    pos_to_aps = [[] for _ in range(n)]
    for ap_id, idx in enumerate(ap_indices):
        for i in idx:
            pos_to_aps[i].append(ap_id)
    
    # Store AP indices as numpy arrays for fast summing
    ap_arrays = [np.array(idx, dtype=np.int32) for idx in ap_indices]
    
    def compute_ap_sums(signs):
        ap_sums = np.zeros(num_aps, dtype=np.int32)
        for ap_id, arr in enumerate(ap_arrays):
            ap_sums[ap_id] = int(np.sum(signs[arr]))
        return ap_sums

    def max_from_sums(ap_sums):
        return int(np.max(np.abs(ap_sums)))

    def build_hist(ap_sums):
        max_hist = n + 1
        hist = np.zeros(max_hist, dtype=np.int32)
        for s in ap_sums:
            hist[abs(s)] += 1
        return hist

    def get_max_from_hist(hist):
        # Find largest k with hist[k] > 0
        for k in range(len(hist) - 1, -1, -1):
            if hist[k] > 0:
                return k
        return 0

    def thue_morse_signs(n):
        """Generate Thue-Morse balanced signs: +1 if popcount(i) even, -1 otherwise."""
        signs = np.zeros(n, dtype=np.int8)
        for i in range(n):
            # Count bits
            cnt = bin(i).count('1')
            signs[i] = 1 if cnt % 2 == 0 else -1
        return signs

    def local_search(signs, max_iters, deadline):
        signs = signs.copy()
        ap_sums = compute_ap_sums(signs)
        hist = build_hist(ap_sums)
        current_max = get_max_from_hist(hist)
        best_signs = signs.copy()
        best_max = current_max
        
        temperature = 3.0
        cooling = 0.9995
        rng = random.Random()
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Occasionally try a 2-flip move to escape local minima
            if rng.random() < 0.15:
                i = rng.randint(0, n - 1)
                j = rng.randint(0, n - 1)
                if i == j:
                    j = (j + 1) % n
                
                old_sign_i = signs[i]
                old_sign_j = signs[j]
                
                # Flip both
                signs[i] = -old_sign_i
                signs[j] = -old_sign_j
                
                affected_i = pos_to_aps[i]
                affected_j = pos_to_aps[j]
                affected = list(set(affected_i + affected_j))
                
                for ap_id in affected:
                    old_abs = abs(ap_sums[ap_id])
                    hist[old_abs] -= 1
                    delta_sum = 0
                    if ap_id in affected_i:
                        delta_sum -= 2 * old_sign_i
                    if ap_id in affected_j:
                        delta_sum -= 2 * old_sign_j
                    ap_sums[ap_id] += delta_sum
                    new_abs = abs(ap_sums[ap_id])
                    hist[new_abs] += 1
                
                new_max = get_max_from_hist(hist)
                delta = new_max - current_max
                
                if delta <= 0 or (temperature > 0.01 and rng.random() < np.exp(-delta / temperature)):
                    current_max = new_max
                    if current_max < best_max:
                        best_max = current_max
                        best_signs = signs.copy()
                else:
                    # Revert
                    signs[i] = old_sign_i
                    signs[j] = old_sign_j
                    for ap_id in affected:
                        old_abs = abs(ap_sums[ap_id])
                        hist[old_abs] -= 1
                        delta_sum = 0
                        if ap_id in affected_i:
                            delta_sum += 2 * old_sign_i
                        if ap_id in affected_j:
                            delta_sum += 2 * old_sign_j
                        ap_sums[ap_id] += delta_sum
                        new_abs = abs(ap_sums[ap_id])
                        hist[new_abs] += 1
            else:
                # Standard single flip
                i = rng.randint(0, n - 1)
                old_sign = signs[i]
                
                # Flip
                signs[i] = -old_sign
                affected = pos_to_aps[i]
                
                # Update histogram and ap_sums
                for ap_id in affected:
                    old_abs = abs(ap_sums[ap_id])
                    hist[old_abs] -= 1
                    ap_sums[ap_id] -= 2 * old_sign
                    new_abs = abs(ap_sums[ap_id])
                    hist[new_abs] += 1
                
                new_max = get_max_from_hist(hist)
                
                delta = new_max - current_max
                
                if delta <= 0 or (temperature > 0.01 and rng.random() < np.exp(-delta / temperature)):
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
                        ap_sums[ap_id] += 2 * old_sign
                        new_abs = abs(ap_sums[ap_id])
                        hist[new_abs] += 1
            
            temperature *= cooling
        
        return best_signs, best_max

    # Multiple restarts with different initializations
    best_overall = None
    best_overall_max = n + 1
    
    # Generate diverse balanced initializations
    initial_signs_list = []
    
    # Thue-Morse
    tm = thue_morse_signs(n)
    initial_signs_list.append(tm)
    
    # Negated Thue-Morse
    initial_signs_list.append(-tm)
    
    # Random balanced sequences with seed diversity
    seeds = [42, 123, 456, 789, 1000, 2024, 3141, 2718, 1618, 9999, 5555, 7777, 1111, 2222, 3333, 4444]
    for seed in seeds:
        rng = random.Random(seed)
        signs = np.array([rng.choice([-1, 1]) for _ in range(n)], dtype=np.int8)
        initial_signs_list.append(signs)
    
    # Also try a few structured initializations
    # Alternating blocks
    for block_size in [4, 8, 16]:
        signs = np.array([1 if (i // block_size) % 2 == 0 else -1 for i in range(n)], dtype=np.int8)
        initial_signs_list.append(signs)
    
    for signs_init in initial_signs_list:
        if time.time() > deadline - 5.0:
            break
        
        # Allocate more iterations to better initializations
        signs, mx = local_search(signs_init, max_iters=3000, deadline=deadline)
        
        if mx < best_overall_max:
            best_overall_max = mx
            best_overall = signs.copy()
            if best_overall_max <= 2:
                break
    
    # Final greedy polish with multiple passes and targeted refinement
    if best_overall is not None:
        signs = best_overall.copy()
        ap_sums = compute_ap_sums(signs)
        hist = build_hist(ap_sums)
        current_max = get_max_from_hist(hist)
        
        # Multiple greedy passes
        for pass_num in range(5):
            if time.time() > deadline - 2.0:
                break
            
            improved = True
            while improved and time.time() < deadline - 2.0:
                improved = False
                # Use different orderings for different passes
                order = list(range(n))
                if pass_num % 2 == 1:
                    order.reverse()
                else:
                    rng_order = random.Random(pass_num * 1000)
                    rng_order.shuffle(order)
                
                for i in order:
                    if time.time() > deadline - 2.0:
                        break
                    old_sign = signs[i]
                    signs[i] = -old_sign
                    for ap_id in pos_to_aps[i]:
                        old_abs = abs(ap_sums[ap_id])
                        hist[old_abs] -= 1
                        ap_sums[ap_id] -= 2 * old_sign
                        new_abs = abs(ap_sums[ap_id])
                        hist[new_abs] += 1
                    new_max = get_max_from_hist(hist)
                    if new_max < current_max:
                        current_max = new_max
                        improved = True
                    else:
                        signs[i] = old_sign
                        for ap_id in pos_to_aps[i]:
                            old_abs = abs(ap_sums[ap_id])
                            hist[old_abs] -= 1
                            ap_sums[ap_id] += 2 * old_sign
                            new_abs = abs(ap_sums[ap_id])
                            hist[new_abs] += 1
        
        # Targeted refinement: focus on positions in the worst APs
        if time.time() < deadline - 1.0:
            for target_pass in range(3):
                if time.time() > deadline - 1.0:
                    break
                
                # Find positions involved in APs with |sum| >= current_max - 1
                target_positions = set()
                for ap_id in range(num_aps):
                    if abs(ap_sums[ap_id]) >= max(0, current_max - 1):
                        for pos in ap_arrays[ap_id]:
                            target_positions.add(pos)
                
                if not target_positions:
                    break
                
                improved = True
                while improved and time.time() < deadline - 1.0:
                    improved = False
                    target_list = list(target_positions)
                    rng_target = random.Random(target_pass * 777)
                    rng_target.shuffle(target_list)
                    
                    for i in target_list:
                        if time.time() > deadline - 1.0:
                            break
                        old_sign = signs[i]
                        signs[i] = -old_sign
                        for ap_id in pos_to_aps[i]:
                            old_abs = abs(ap_sums[ap_id])
                            hist[old_abs] -= 1
                            ap_sums[ap_id] -= 2 * old_sign
                            new_abs = abs(ap_sums[ap_id])
                            hist[new_abs] += 1
                        new_max = get_max_from_hist(hist)
                        if new_max < current_max:
                            current_max = new_max
                            improved = True
                        else:
                            signs[i] = old_sign
                            for ap_id in pos_to_aps[i]:
                                old_abs = abs(ap_sums[ap_id])
                                hist[old_abs] -= 1
                                ap_sums[ap_id] += 2 * old_sign
                                new_abs = abs(ap_sums[ap_id])
                                hist[new_abs] += 1
        
        if current_max < best_overall_max:
            best_overall_max = current_max
            best_overall = signs.copy()
    
    if best_overall is None:
        rng = random.Random(42)
        return [rng.choice([-1, 1]) for _ in range(n)]
    
    # Final verification and cleanup
    final_signs = [int(s) for s in best_overall]
    assert len(final_signs) == n
    assert all(s in [-1, 1] for s in final_signs)
    
    return final_signs
# EVOLVE-BLOCK-END
