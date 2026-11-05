import time
import sys
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s limit for safety

    denominators = list(range(2, 49))
    n = len(denominators)
    
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a
        
    def lcm(a, b):
        return a // gcd(a, b) * b
        
    L = 1
    for d in denominators:
        L = lcm(L, d)
        
    weights = [L // d for d in denominators]
    
    items = [(weights[i], i) for i in range(n)]
    items.sort(key=lambda x: x[0], reverse=True)
    
    sorted_weights = [item[0] for item in items]
    sorted_to_orig = [item[1] for item in items]
    
    suffix_sums = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sums[i] = suffix_sums[i + 1] + sorted_weights[i]
        
    best_sum = 0
    best_mask = 0
    
    # Greedy Seed
    curr_sum = 0
    curr_mask = 0
    for i in range(n):
        if curr_sum + sorted_weights[i] < L:
            curr_sum += sorted_weights[i]
            curr_mask |= (1 << i)
    if curr_sum > best_sum:
        best_sum = curr_sum
        best_mask = curr_mask

    # B&B
    def dfs(idx, curr_sum, curr_mask):
        nonlocal best_sum, best_mask
        
        if time.time() > deadline:
            return
            
        if curr_sum + suffix_sums[idx] <= best_sum:
            return
            
        if idx == n:
            if curr_sum > best_sum:
                best_sum = curr_sum
                best_mask = curr_mask
            return
            
        if curr_sum + sorted_weights[idx] < L:
            dfs(idx + 1, curr_sum + sorted_weights[idx], curr_mask | (1 << idx))
            
        dfs(idx + 1, curr_sum, curr_mask)

    dfs(0, 0, 0)
    
    selected_orig_indices = set()
    for i in range(n):
        if best_mask & (1 << i):
            selected_orig_indices.add(sorted_to_orig[i])
            
    if not selected_orig_indices:
        return [0, 1, 5]

    total_frac = sum(Fraction(1, denominators[k]) for k in selected_orig_indices)
    if total_frac >= 1:
        return [0, 1, 5]
        
    return sorted(selected_orig_indices)
