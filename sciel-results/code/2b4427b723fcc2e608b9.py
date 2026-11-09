import time
from math import gcd
from fractions import Fraction
from bisect import bisect_left

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Calculate LCM
    L = 1
    for d in denominators:
        L = L * d // gcd(L, d)
        
    weights = [L // d for d in denominators]
    
    # Split indices
    mid = n // 2
    idx1 = list(range(mid))
    idx2 = list(range(mid, n))
    
    w1 = [weights[i] for i in idx1]
    w2 = [weights[i] for i in idx2]
    
    # Generate all subset sums for half 1
    # Store (sum, mask)
    sums1 = [(0, 0)]
    for i in range(len(w1)):
        wi = w1[i]
        # Use list comprehension to generate new sums
        # This is fast in CPython
        sums1 += [(s + wi, m | (1 << i)) for s, m in sums1]
        
    # Sort by sum
    sums1.sort(key=lambda x: x[0])
    
    # Extract sums for binary search
    sums1_values = [s for s, m in sums1]
    
    # Precompute suffix sums for w2 for pruning
    suffix_sums = [0] * (len(w2) + 1)
    for i in range(len(w2) - 1, -1, -1):
        suffix_sums[i] = suffix_sums[i + 1] + w2[i]
        
    max_s1 = sums1_values[-1] if sums1_values else 0
    
    best_deficit = L # Max possible deficit
    best_mask1 = 0
    best_mask2 = 0
    
    # DFS for half 2 with pruning
    def dfs2(pos, current_sum, current_mask):
        nonlocal best_deficit, best_mask1, best_mask2
        
        if time.time() > deadline:
            return
            
        # Pruning: Check if even taking all remaining weights in w2 
        # and the best possible s1 cannot beat the current best
        max_possible_total = current_sum + suffix_sums[pos] + max_s1
        if max_possible_total < L - best_deficit:
            return

        if pos == len(w2):
            if current_sum < L:
                # Find largest s1 such that s1 + current_sum < L
                # i.e., s1 < L - current_sum
                target = L - current_sum
                # bisect_left gives first index >= target
                # We want largest < target, so index - 1
                idx = bisect_left(sums1_values, target)
                if idx > 0:
                    s1_val, m1 = sums1[idx - 1]
                    total_sum = s1_val + current_sum
                    if total_sum < L:
                        deficit = L - total_sum
                        if deficit < best_deficit:
                            best_deficit = deficit
                            best_mask1 = m1
                            best_mask2 = current_mask
            return
        
        # Option 1: Include w2[pos] (Explore larger sums first to tighten bound)
        new_sum = current_sum + w2[pos]
        if new_sum < L:
            dfs2(pos + 1, new_sum, current_mask | (1 << pos))
            
        # Option 2: Exclude w2[pos]
        dfs2(pos + 1, current_sum, current_mask)
        
    dfs2(0, 0, 0)
    
    # Reconstruct indices
    result_indices = []
    for i in range(len(idx1)):
        if best_mask1 & (1 << i):
            result_indices.append(idx1[i])
            
    for i in range(len(idx2)):
        if best_mask2 & (1 << i):
            result_indices.append(idx2[i])
            
    if not result_indices:
        return [0]
        
    return result_indices
