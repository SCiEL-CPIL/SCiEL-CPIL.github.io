import time
from math import lcm

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 85.0  # 85 seconds to be safe against 90s limit
    
    n = 30
    # Calculate LCM of 1..30
    L = 1
    for i in range(1, n + 1):
        L = lcm(L, i)
    
    # Precompute integer weights w_i = L / i
    weights = [L // i for i in range(1, n + 1)]
    
    # Split into two halves for Meet-in-the-Middle
    half_size = n // 2
    w1 = weights[:half_size]
    w2 = weights[half_size:]
    
    def generate_sums(w_list):
        sums = [0]
        for w in w_list:
            # List comprehension is faster
            sums = [s - w for s in sums] + sums + [s + w for s in sums]
        return sums

    def find_coeffs_for_sum(weights, target_sum):
        n_w = len(weights)
        suffix_max = [0] * (n_w + 1)
        for i in range(n_w - 1, -1, -1):
            suffix_max[i] = suffix_max[i+1] + weights[i]
            
        def dfs(idx, current_sum):
            if idx == n_w:
                if current_sum == target_sum:
                    return []
                return None
            
            # Pruning
            if current_sum + suffix_max[idx] < target_sum:
                return None
            if current_sum - suffix_max[idx] > target_sum:
                return None
            
            # Heuristic: Try 0 first, then +/-1
            res = dfs(idx + 1, current_sum)
            if res is not None:
                return [0] + res
                
            res = dfs(idx + 1, current_sum + weights[idx])
            if res is not None:
                return [1] + res
                
            res = dfs(idx + 1, current_sum - weights[idx])
            if res is not None:
                return [-1] + res
                
            return None
        
        return dfs(0, 0)

    # Generate sums for both halves
    sums1 = generate_sums(w1)
    sums2 = generate_sums(w2)
    
    # Sort both lists for Two-Pointer approach
    sums1.sort()
    sums2.sort()
    
    best_abs_int_diff = float('inf')
    best_s1 = 0
    best_s2 = 0
    
    n1 = len(sums1)
    n2 = len(sums2)
    
    i = 0
    j = n2 - 1
    
    while i < n1 and j >= 0:
        if time.time() - start_time > deadline - 5:
            break
            
        s_val = sums1[i] + sums2[j]
        
        if s_val == 0:
            # Need non-zero. Move to next candidate.
            # We can move i or j. Moving i is standard.
            i += 1
            continue
            
        abs_val = abs(s_val)
        if abs_val < best_abs_int_diff:
            best_abs_int_diff = abs_val
            best_s1 = sums1[i]
            best_s2 = sums2[j]
            if best_abs_int_diff == 1:
                break
                
        if s_val > 0:
            j -= 1
        else:
            i += 1
            
    if best_abs_int_diff == float('inf'):
        # Fallback
        return [0] * 28 + [1, -1]
        
    # Reconstruct coefficients
    c1 = find_coeffs_for_sum(w1, best_s1)
    c2 = find_coeffs_for_sum(w2, best_s2)
    
    if c1 is None or c2 is None:
        return [0] * 28 + [1, -1]
        
    result = c1 + c2
    
    # Verify
    total = sum(c * w for c, w in zip(result, weights))
    if total == 0:
        return [0] * 28 + [1, -1]
        
    return result

# EVOLVE-BLOCK-END
