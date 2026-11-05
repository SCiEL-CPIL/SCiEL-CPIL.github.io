import time
import numpy as np
from math import lcm

def construct():
    start_time = time.time()
    deadline = 80.0  # Safety margin before 90s limit
    
    n = 30
    # LCM of 1..30
    L = 1
    for i in range(1, n + 1):
        L = lcm(L, i)
    
    # Integer weights: w_i = L // (i+1) for coefficient c_i multiplying 1/(i+1)
    # Sum = sum(c_i * L/(i+1)) / L
    # Minimize |sum(c_i * w_i)| where w_i = L//(i+1)
    weights = np.array([L // (i + 1) for i in range(n)], dtype=np.int64)
    
    # Split into two halves of 15
    half1 = weights[:15]
    half2 = weights[15:]
    
    # Generate all possible integer sums for each half
    # Use iterative approach with numpy for speed
    
    def gen_sums(w):
        """Generate all 3^len(w) possible sums using numpy"""
        sums = np.array([0], dtype=np.int64)
        for wi in w:
            # For each existing sum, create sum-wi, sum, sum+wi
            new_sums = np.concatenate([sums - wi, sums, sums + wi])
            sums = new_sums
        return sums
    
    # Generate sums for both halves
    sums1 = gen_sums(half1)
    sums2 = gen_sums(half2)
    
    # Sort sums2 for efficient search
    sums2_sorted = np.sort(sums2)
    sums1_sorted = np.sort(sums1)
    
    # Two-pointer technique on sorted arrays to find min |s1 + s2| != 0
    # We want sums1[i] + sums2[j] close to 0 but not 0
    min_abs = np.iinfo(np.int64).max
    best_pairs = [] # Store top K pairs
    
    i = 0
    j = len(sums2_sorted) - 1
    n1 = len(sums1_sorted)
    n2 = len(sums2_sorted)
    
    while i < n1 and j >= 0:
        s_val = sums1_sorted[i] + sums2_sorted[j]
        
        if s_val != 0:
            abs_val = abs(s_val)
            if abs_val < min_abs:
                min_abs = abs_val
                best_pairs = [(sums1_sorted[i], sums2_sorted[j])]
                if min_abs == 1:
                    break
        
        if s_val > 0:
            j -= 1
        elif s_val < 0:
            i += 1
        else:
            # s_val == 0: check neighbors to find best nonzero
            candidates = []
            if i + 1 < n1:
                cv = sums1_sorted[i + 1] + sums2_sorted[j]
                if cv != 0:
                    candidates.append((abs(cv), sums1_sorted[i + 1], sums2_sorted[j]))
            if j - 1 >= 0:
                cv = sums1_sorted[i] + sums2_sorted[j - 1]
                if cv != 0:
                    candidates.append((abs(cv), sums1_sorted[i], sums2_sorted[j - 1]))
            if candidates:
                best_c = min(candidates)
                if best_c[0] < min_abs:
                    min_abs = best_c[0]
                    best_pairs = [(best_c[1], best_c[2])]
                    if min_abs == 1:
                        break
            i += 1
            j -= 1
            
    if min_abs == np.iinfo(np.int64).max:
        # Fallback: return a simple nonzero solution
        fallback = [0] * 28 + [1, -1]
        return fallback
        
    best_s1, best_s2 = best_pairs[0]

    # Now recover coefficients for best_s1 using half1 weights
    # and best_s2 using half2 weights
    # Use DFS with pruning
    
    def find_coeffs(weights_list, target):
        """Find coefficients in {-1,0,1} such that sum(c_i * w_i) == target"""
        n_w = len(weights_list)
        # Precompute suffix sums for pruning
        suffix_max = [0] * (n_w + 1)
        for k in range(n_w - 1, -1, -1):
            suffix_max[k] = suffix_max[k + 1] + abs(weights_list[k])
        
        # Sort weights by magnitude descending to improve pruning?
        # The input is already descending for half1 and half2.
        # But let's ensure it.
        indexed_weights = sorted(enumerate(weights_list), key=lambda x: abs(x[1]), reverse=True)
        sorted_weights = [w for _, w in indexed_weights]
        # We need to map back to original indices if we change order.
        # But the output is just a list of coefficients for the *original* order?
        # No, the result is `c1 + c2`.
        # `c1` corresponds to `half1`.
        # If we sort `half1` for DFS, we get coefficients in the *sorted* order.
        # We need to unsort them.
        
        # However, `half1` is already $L/1, \dots, L/15$ which is descending.
        # So `indexed_weights` will be in the same order.
        # So we can just use the list as is.
        
        def dfs(idx, cur_sum, coeffs):
            if time.time() > deadline:
                return None
            if idx == n_w:
                if cur_sum == target:
                    return list(coeffs)
                return None
            
            rem = suffix_max[idx]
            # Prune
            if cur_sum + rem < target or cur_sum - rem > target:
                return None
            
            w = weights_list[idx]
            # Try -1
            coeffs.append(-1)
            res = dfs(idx + 1, cur_sum - w, coeffs)
            if res is not None:
                return res
            coeffs.pop()
            
            # Try 0
            coeffs.append(0)
            res = dfs(idx + 1, cur_sum, coeffs)
            if res is not None:
                return res
            coeffs.pop()
            
            # Try 1
            coeffs.append(1)
            res = dfs(idx + 1, cur_sum + w, coeffs)
            if res is not None:
                return res
            coeffs.pop()
            
            return None
        
        return dfs(0, 0, [])
    
    c1 = find_coeffs(list(half1), int(best_s1))
    c2 = find_coeffs(list(half2), int(best_s2))
    
    if c1 is None or c2 is None:
        # Fallback
        fallback = [0] * 28 + [1, -1]
        return fallback
    
    result = c1 + c2
    
    if time.time() > deadline:
        return fallback
    
    # Verify the result
    total_int = sum(c * w for c, w in zip(result, weights))
    if total_int == 0:
        # Should not happen, but fallback
        fallback = [0] * 28 + [1, -1]
        return fallback
        
    return result
