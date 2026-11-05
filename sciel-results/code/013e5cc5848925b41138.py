import time
from math import lcm
from fractions import Fraction
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 80.0  # 80 seconds to be safe against 90s limit
    
    n = 30
    # Calculate LCM of 1..30
    L = 1
    for i in range(1, n + 1):
        L = lcm(L, i)
    
    # Precompute integer weights w_i = L / i
    weights = [L // i for i in range(1, n + 1)]
    
    # Split into two halves
    half1_weights = weights[:15]
    half2_weights = weights[15:]
    
    def generate_sums_np(w_list):
        """Generate all possible sums using NumPy for speed."""
        sums = np.array([0], dtype=np.int64)
        for w in w_list:
            w = int(w)
            new_sums = np.concatenate([sums - w, sums, sums + w])
            sums = new_sums
        return sums
    
    # Check time
    if time.time() - start_time > deadline - 5:
        return [0] * 28 + [1, -1]
    
    t0 = time.time()
    sums1 = generate_sums_np(half1_weights)
    t1 = time.time()
    
    # For half 2, we need to store coefficients for reconstruction
    # Use a dict: sum -> one example coefficient tuple
    # This is memory heavy but necessary for reconstruction
    
    def generate_sums_with_coeffs(w_list):
        """Generate sums and store one coefficient set per sum value."""
        # Start with {(0,): (0,)}
        sums_dict = {0: (0,)}
        for w in w_list:
            new_dict = {}
            for s, coeffs in sums_dict.items():
                # Try -1
                new_s = s - w
                if new_s not in new_dict:
                    new_dict[new_s] = coeffs + (-1,)
                # Try 0
                if s not in new_dict:
                    new_dict[s] = coeffs + (0,)
                # Try 1
                new_s2 = s + w
                if new_s2 not in new_dict:
                    new_dict[new_s2] = coeffs + (1,)
            sums_dict = new_dict
        return sums_dict
    
    t2 = time.time()
    # Generate sums2 with coefficients stored
    sums2_dict = generate_sums_with_coeffs(half2_weights)
    t3 = time.time()
    
    # Convert to sorted list for two-pointer
    sums2 = np.array(sorted(sums2_dict.keys()), dtype=np.int64)
    
    t4 = time.time()
    sums1_sorted = np.sort(sums1)
    t5 = time.time()
    
    # Two-pointer search for min |a + b| != 0
    i = 0
    j = len(sums2) - 1
    min_abs = float('inf')
    best_a = 0
    best_b = 0
    
    len1 = len(sums1_sorted)
    len2 = len(sums2)
    
    while i < len1 and j >= 0:
        s_val = int(sums1_sorted[i]) + int(sums2[j])
        if s_val != 0:
            abs_val = abs(s_val)
            if abs_val < min_abs:
                min_abs = abs_val
                best_a = int(sums1_sorted[i])
                best_b = int(sums2[j])
                if min_abs == 1:
                    break
        
        if s_val > 0:
            j -= 1
        elif s_val < 0:
            i += 1
        else:
            i += 1
            j -= 1
        
        # Time check
        if time.time() - start_time > deadline - 3:
            break
    
    if min_abs == float('inf'):
        return [0] * 28 + [1, -1]
        
    # Now we have best_a (sum for half 1) and best_b (sum for half 2).
    # We need to find coefficients for best_a using half1_weights.
    # For half 2, we have it in sums2_dict.
    
    # For half 1, use DFS with pruning
    def find_coeffs_for_sum(weights, target_sum):
        n_w = len(weights)
        suffix_max = [0] * (n_w + 1)
        for i in range(n_w - 1, -1, -1):
            suffix_max[i] = suffix_max[i+1] + weights[i]
            
        def dfs(idx, current_sum, path):
            if idx == n_w:
                if current_sum == target_sum:
                    return list(path)
                return None
            
            remaining_max = suffix_max[idx]
            if current_sum + remaining_max < target_sum or current_sum - remaining_max > target_sum:
                return None
                
            # Try -1
            path.append(-1)
            res = dfs(idx + 1, current_sum - weights[idx], path)
            if res is not None:
                return res
            path.pop()
            
            # Try 0
            path.append(0)
            res = dfs(idx + 1, current_sum, path)
            if res is not None:
                return res
            path.pop()
            
            # Try 1
            path.append(1)
            res = dfs(idx + 1, current_sum + weights[idx], path)
            if res is not None:
                return res
            path.pop()
            
            return None
        
        return dfs(0, 0, [])

    c1 = find_coeffs_for_sum(half1_weights, best_a)
    
    # For half 2, get coefficients from dict
    if best_b in sums2_dict:
        c2 = list(sums2_dict[best_b])
    else:
        # Fallback: should not happen
        c2 = find_coeffs_for_sum(half2_weights, best_b)
    
    if c1 is None or c2 is None:
        return [0] * 28 + [1, -1]
        
    result = c1 + c2
    
    # Verify
    total = sum(c * w for c, w in zip(result, weights))
    if total == 0:
        return [0] * 28 + [1, -1]
        
    return result

# EVOLVE-BLOCK-END
