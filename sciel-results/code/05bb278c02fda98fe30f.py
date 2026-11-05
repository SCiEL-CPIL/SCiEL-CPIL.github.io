import time
from fractions import Fraction

def construct():
    start = time.time()
    deadline = start + 85  # 85 second budget to be safe
    
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Calculate LCM of 2..48
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a
            
    def lcm(a, b):
        return a // gcd(a, b) * b
        
    L = 1
    for d in denominators:
        L = lcm(L, d)
        
    # Weights are integers w_i = L / d_i
    # Sort denominators ascending -> weights descending
    # This helps pruning because large weights are considered first
    pairs = sorted([(d, L // d) for d in denominators], key=lambda x: x[1], reverse=True)
    weights = [w for d, w in pairs]
    
    # Precompute suffix sums
    suffix = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix[i] = suffix[i + 1] + weights[i]
        
    # Greedy initialization to set a good initial bound
    best_sum = 0
    best_mask = 0
    greedy_sum = 0
    greedy_mask = 0
    for i in range(n):
        if greedy_sum + weights[i] < L:
            greedy_sum += weights[i]
            greedy_mask |= (1 << i)
    
    best_sum = greedy_sum
    best_mask = greedy_mask
    
    # DFS
    # idx: current index
    # curr_sum: current sum
    # mask: bitmask of selected items
    def dfs(idx, curr_sum, mask):
        nonlocal best_sum, best_mask
        
        # Time check
        if time.time() > deadline - 5:
            return
        
        # Pruning 1: Even if we take all remaining, we can't beat best
        if curr_sum + suffix[idx] <= best_sum:
            return
            
        # Pruning 2: Sum exceeds or equals L
        if curr_sum >= L:
            return
            
        if idx == n:
            if curr_sum > best_sum:
                best_sum = curr_sum
                best_mask = mask
            return
            
        # Branch 1: Take item idx
        # Check if taking it is valid
        new_sum = curr_sum + weights[idx]
        if new_sum < L:
            dfs(idx + 1, new_sum, mask | (1 << idx))
            
        # Branch 2: Skip item idx
        dfs(idx + 1, curr_sum, mask)
        
    dfs(0, 0, 0)
    
    # Reconstruct indices
    # We need to map back to original zero-based indices in [2, ..., 48]
    # pairs[i] = (denominator, weight)
    # Original index for denominator d is d - 2
    selected_original_indices = []
    for i in range(n):
        if best_mask & (1 << i):
            d = pairs[i][0]
            selected_original_indices.append(d - 2)
            
    # Verify
    if not selected_original_indices:
        # Fallback
        return [0, 1, 5]
        
    total_frac = sum(Fraction(1, denominators[k]) for k in selected_original_indices)
    if total_frac < 1:
        return sorted(selected_original_indices)
    else:
        return [0, 1, 5]
