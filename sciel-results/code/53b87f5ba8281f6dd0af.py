import time
from math import lcm
from fractions import Fraction

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
    
    # We want to find c_i in {-1, 0, 1} such that S = sum(c_i * w_i) is nonzero and |S| is minimized.
    # The actual value is S / L.
    
    # Split into two halves for Meet-in-the-Middle
    half_size = n // 2
    w1 = weights[:half_size]
    w2 = weights[half_size:]
    
    # Generate all possible sums for the first half
    # We store (sum, coeffs_tuple)
    # To save memory and time, we can just generate sums and store them in a list.
    # But we need coeffs to return.
    # 3^15 = 14,348,907.
    # Storing 14M tuples of 15 ints is heavy.
    # Alternative: Store just the sums in a sorted list.
    # Then, once we find the optimal pair (s1, s2), we reconstruct the coeffs.
    
    def generate_sums(w_list):
        sums = [0]
        for w in w_list:
            new_sums = []
            # Optimization: pre-allocate or use list comprehension?
            # List comprehension is faster in Python.
            # But we are doing 3 operations per element.
            for s in sums:
                new_sums.append(s - w)
                new_sums.append(s)
                new_sums.append(s + w)
            sums = new_sums
        return sums

    def find_coeffs_for_sum(weights, target_sum):
        # Backtracking search to find coefficients for a specific target sum
        # weights length is 15.
        n_w = len(weights)
        # Sort weights and track indices? No, we need to map back to original positions.
        # Let's just do DFS on the original order.
        # Pruning:
        # Suffix sums of absolute weights?
        # Max possible sum from remaining: sum(remaining_weights)
        # Min possible sum from remaining: -sum(remaining_weights)
        
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
            
            # Try -1, 0, 1
            # Heuristic: try 0 first? Or try the one that gets closest?
            # Let's try 0 first, then +/-1.
            
            # Actually, checking distance to target might help.
            # dist_0 = abs(target_sum - current_sum)
            # dist_1 = abs(target_sum - (current_sum + weights[idx]))
            # dist_-1 = abs(target_sum - (current_sum - weights[idx]))
            # Sort options by distance?
            
            # For simplicity and speed, just try 0, 1, -1.
            
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
    t0 = time.time()
    sums1 = generate_sums(w1)
    sums2 = generate_sums(w2)
    
    # Sort sums2 for binary search
    sums2.sort()
    
    best_abs_int_diff = float('inf')
    best_s1 = 0
    best_s2 = 0
    
    # Iterate through sums1
    # For each s1, we want s2 such that |s1 + s2| is minimized and non-zero.
    # Binary search in sums2 for -s1.
    
    len2 = len(sums2)
    
    for s1 in sums1:
        if time.time() - start_time > deadline - 10:
            break
            
        target = -s1
        # Binary search for target in sums2
        # We want the element closest to target.
        # bisect module is fast.
        
        # We can use bisect.bisect_left
        import bisect
        idx = bisect.bisect_left(sums2, target)
        
        # Check idx and idx-1
        candidates = []
        if idx < len2:
            candidates.append(sums2[idx])
        if idx > 0:
            candidates.append(sums2[idx-1])
            
        for s2 in candidates:
            diff = s1 + s2
            if diff != 0:
                abs_diff = abs(diff)
                if abs_diff < best_abs_int_diff:
                    best_abs_int_diff = abs_diff
                    best_s1 = s1
                    best_s2 = s2
                    if best_abs_int_diff == 1:
                        break
        if best_abs_int_diff == 1:
            break
            
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
