import math
import time
import random
import numpy as np
from math import gcd
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s safety margin for 90s kill

    N = 30
    # Calculate LCM of 1..30
    L = 1
    for i in range(1, N + 1):
        L = L * i // gcd(L, i)

    # Precompute weights w_i = L // i
    # weights[0] corresponds to i=1, ..., weights[29] to i=30
    weights = np.array([L // i for i in range(1, N + 1)], dtype=np.int64)

    def calc_sum(coeffs):
        s = 0
        for i in range(N):
            if coeffs[i] != 0:
                s += coeffs[i] * weights[i]
        return s

    best_abs_num = float('inf')
    best_coeffs = [1] + [0] * (N - 1)  # Fallback: sum = 1

    # Phase 1: Sparse Search (from Current Program)
    # We look for subsets of indices of size k, and sign assignments, such that |sum| is minimized and non-zero.
    # This is very effective for finding small numerators (1, 2, 3...) because sparse cancellations are easier to find.
    # k=2, 3, 4 are fast. k=5 might be slow but we'll try if time permits.
    
    max_k = 4
    for k in range(2, max_k + 1):
        if time.time() > deadline - 15:
            break

        # Iterate over all combinations of k indices
        for combo in combinations(range(N), k):
            # Precompute weights for this combo
            w_combo = [weights[i] for i in combo]

            # Enumerate all 2^k sign patterns
            for mask in range(1 << k):
                s = 0
                for j in range(k):
                    if (mask >> j) & 1:
                        s -= w_combo[j]
                    else:
                        s += w_combo[j]

                if s != 0:
                    abs_s = abs(s)
                    if abs_s < best_abs_num:
                        best_abs_num = abs_s
                        coeffs = [0] * N
                        for j in range(k):
                            idx = combo[j]
                            if (mask >> j) & 1:
                                coeffs[idx] = -1
                            else:
                                coeffs[idx] = 1
                        best_coeffs = coeffs

                        if best_abs_num == 1:
                            return best_coeffs

    # Phase 2: Meet-in-the-Middle (from Inspiration 1)
    # This explores the full space more efficiently than brute force.
    # Split weights into two halves.
    # Generate all possible sums for each half.
    # Find pairs (s1, s2) such that |s1 + s2| is minimized and non-zero.
    
    if time.time() < deadline - 20:
        mid = N // 2
        w1 = weights[:mid]
        w2 = weights[mid:]

        def generate_sums(weights_half):
            sums = np.array([0], dtype=np.int64)
            for w in weights_half:
                sums = np.concatenate([sums - w, sums, sums + w])
            return sums

        try:
            sums1 = generate_sums(w1)
            sums2 = generate_sums(w2)

            # Sort sums1 for binary search
            sums1_sorted = np.sort(sums1)
            # We need to keep track of original indices? No, we just need the value to reconstruct later?
            # Actually, reconstruction is the hard part.
            # Inspiration 1 uses a DFS to reconstruct the coefficients for a specific sum value.
            # Let's find the best (s1, s2) pair first.

            targets = -sums2
            len_s1 = len(sums1_sorted)

            # Find predecessor and successor indices in sums1_sorted for each target
            idx_l = np.searchsorted(sums1_sorted, targets, side='left')
            idx_r = np.searchsorted(sums1_sorted, targets, side='right')

            # Candidate 1: Predecessor
            idx_cand1 = idx_l - 1
            valid1 = idx_cand1 >= 0
            idx_cand1_clipped = np.clip(idx_cand1, 0, len_s1 - 1)
            s1_cand1 = sums1_sorted[idx_cand1_clipped]

            total1 = sums2 + s1_cand1
            abs_total1 = np.abs(total1)

            # Mask out invalid and zero sums
            mask1 = valid1 & (total1 != 0)
            abs_total1[~mask1] = np.iinfo(np.int64).max

            # Candidate 2: Successor
            idx_cand2 = idx_r
            valid2 = idx_cand2 < len_s1
            idx_cand2_clipped = np.clip(idx_cand2, 0, len_s1 - 1)
            s1_cand2 = sums1_sorted[idx_cand2_clipped]

            total2 = sums2 + s1_cand2
            abs_total2 = np.abs(total2)

            # Mask out invalid and zero sums
            mask2 = valid2 & (total2 != 0)
            abs_total2[~mask2] = np.iinfo(np.int64).max

            # Find global minimum
            min_abs1 = np.min(abs_total1)
            min_abs2 = np.min(abs_total2)

            if min_abs1 < min_abs2:
                best_abs_mim = min_abs1
                idx_best_in_1 = np.argmin(abs_total1)
                s2_val = sums2[idx_best_in_1]
                s1_val = s1_cand1[idx_best_in_1]
            else:
                best_abs_mim = min_abs2
                idx_best_in_2 = np.argmin(abs_total2)
                s2_val = sums2[idx_best_in_2]
                s1_val = s1_cand2[idx_best_in_2]

            # Reconstruct coefficients
            # We need to find coeffs c1 for w1 such that sum(c1[i]*w1[i]) == s1_val
            # and coeffs c2 for w2 such that sum(c2[i]*w2[i]) == s2_val
            
            def reconstruct_coeffs(weights, target_sum):
                n_w = len(weights)
                # Sort weights by magnitude descending for better pruning
                order = np.argsort(-weights)
                sorted_weights = weights[order]
                
                # Suffix sums for pruning
                suffix_max = np.zeros(n_w + 1, dtype=np.int64)
                suffix_min = np.zeros(n_w + 1, dtype=np.int64)
                for i in range(n_w - 1, -1, -1):
                    suffix_max[i] = suffix_max[i+1] + sorted_weights[i]
                    suffix_min[i] = suffix_min[i+1] - sorted_weights[i]
                    
                coeffs = [0] * n_w
                
                def dfs(i, current_sum):
                    if i == n_w:
                        return current_sum == target_sum
                    
                    # Pruning
                    if current_sum + suffix_max[i] < target_sum:
                        return False
                    if current_sum + suffix_min[i] > target_sum:
                        return False
                        
                    # Try -1
                    coeffs[i] = -1
                    if dfs(i + 1, current_sum - sorted_weights[i]):
                        return True
                    # Try 0
                    coeffs[i] = 0
                    if dfs(i + 1, current_sum):
                        return True
                    # Try 1
                    coeffs[i] = 1
                    if dfs(i + 1, current_sum + sorted_weights[i]):
                        return True
                        
                    return False
                
                if dfs(0, 0):
                    orig_coeffs = [0] * n_w
                    for i in range(n_w):
                        orig_idx = order[i]
                        orig_coeffs[orig_idx] = coeffs[i]
                    return orig_coeffs
                else:
                    return None

            c1 = reconstruct_coeffs(w1, s1_val)
            c2 = reconstruct_coeffs(w2, s2_val)
            
            if c1 is not None and c2 is not None:
                final_coeffs = c1 + c2
                # Verify
                val = calc_sum(final_coeffs)
                if val != 0 and abs(val) < best_abs_num:
                    best_abs_num = abs(val)
                    best_coeffs = final_coeffs
                    if best_abs_num == 1:
                        return best_coeffs

        except Exception:
            pass # Fallback to local search if MIM fails or is too slow

    # Phase 3: Local Search Refinement
    # Use the best found so far as a starting point for local search.
    # This can often squeeze out a better solution if the sparse or MIM solution is not optimal.
    
    current_coeffs = best_coeffs[:]
    current_sum = calc_sum(current_coeffs)
    current_abs = abs(current_sum)
    
    random.seed(42)
    
    # Try a few random restarts if time permits
    num_restarts = 10
    for r in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        if r == 0:
            c = best_coeffs[:]
        else:
            c = [random.choice([-1, 0, 1]) for _ in range(N)]
        
        s = calc_sum(c)
        if s == 0:
            i = random.randint(0, N-1)
            c[i] = 1 if c[i] != 1 else -1
            s = calc_sum(c)
            if s == 0:
                continue
        
        abs_s = abs(s)
        
        # Local Search: First Improvement
        improved = True
        steps = 0
        max_steps = 500
        while improved and steps < max_steps and time.time() < deadline - 2:
            improved = False
            steps += 1
            for i in range(N):
                old_val = c[i]
                candidates = []
                if old_val == -1:
                    candidates = [0, 1]
                elif old_val == 0:
                    candidates = [-1, 1]
                else:
                    candidates = [-1, 0]
                    
                for new_val in candidates:
                    delta = (new_val - old_val) * weights[i]
                    s_new = s + delta
                    
                    if s_new != 0:
                        abs_new = abs(s_new)
                        if abs_new < abs_s:
                            c[i] = new_val
                            s = s_new
                            abs_s = abs_new
                            improved = True
                            
                            if abs_s == 1:
                                return c
                            
                            break
                if improved:
                    break
        
        if abs_s < best_abs_num:
            best_abs_num = abs_s
            best_coeffs = c[:]
            if best_abs_num == 1:
                return best_coeffs

    return best_coeffs
