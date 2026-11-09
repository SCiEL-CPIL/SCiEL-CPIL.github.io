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
    
    # Split: put larger weights in DFS side for better pruning
    # Sort by weight descending, then split
    sorted_indices = sorted(range(n), key=lambda i: weights[i], reverse=True)
    
    # Put the largest 23 in DFS (side 2), smallest 24 in enumeration (side 1)
    # This way DFS hits large sums first
    mid = n - 23  # 24 in side1, 23 in side2
    idx1 = sorted_indices[:mid]      # smaller weights, enumerate all 2^24
    idx2 = sorted_indices[mid:]      # larger weights, DFS with 2^23
    
    w1 = [weights[i] for i in idx1]
    w2 = [weights[i] for i in idx2]
    
    # Generate all subset sums for half 1 (smaller weights, 2^24 = 16.7M - too many!)
    # Actually 2^24 is too large. Let me reconsider.
    
    # Better: split 23/24, enumerate the smaller side
    # 2^23 = 8.4M is manageable for enumeration
    # Let's put 23 in enumeration, 24 in DFS
    # But we want large weights in DFS for pruning...
    
    # Alternative: keep original split but optimize
    mid = n // 2  # 23
    idx1 = list(range(mid))
    idx2 = list(range(mid, n))
    
    w1 = [weights[i] for i in idx1]
    w2 = [weights[i] for i in idx2]
    
    # Sort w2 descending for better pruning in DFS
    w2_order = sorted(range(len(w2)), key=lambda i: w2[i], reverse=True)
    w2_sorted = [w2[i] for i in w2_order]
    
    # Precompute suffix sums for pruning
    suffix_max = [0] * (len(w2_sorted) + 1)
    for i in range(len(w2_sorted) - 1, -1, -1):
        suffix_max[i] = suffix_max[i + 1] + w2_sorted[i]
    
    # Generate all subset sums for half 1
    sums1 = [(0, 0)]
    for i in range(len(w1)):
        wi = w1[i]
        sums1 += [(s + wi, m | (1 << i)) for s, m in sums1]
        
    sums1.sort(key=lambda x: x[0])
    sums1_values = [s for s, m in sums1]
    
    best_deficit = L
    best_mask1 = 0
    best_mask2 = 0
    best_total_sum = 0
    
    def dfs2(pos, current_sum, current_mask):
        nonlocal best_deficit, best_mask1, best_mask2, best_total_sum
        
        if time.time() > deadline:
            return
        
        # Pruning: if current_sum + suffix_max[pos] <= best_total_sum, can't improve
        if current_sum + suffix_max[pos] <= best_total_sum:
            return
        
        if pos == len(w2_sorted):
            if current_sum < L:
                target = L - current_sum
                idx = bisect_left(sums1_values, target)
                if idx > 0:
                    s1_val, m1 = sums1[idx - 1]
                    total_sum = s1_val + current_sum
                    if total_sum < L and total_sum > best_total_sum:
                        best_deficit = L - total_sum
                        best_mask1 = m1
                        best_mask2 = current_mask
                        best_total_sum = total_sum
            return
        
        # Include first (large elements first for better pruning)
        new_sum = current_sum + w2_sorted[pos]
        if new_sum < L:
            dfs2(pos + 1, new_sum, current_mask | (1 << w2_order[pos]))
        
        # Exclude
        dfs2(pos + 1, current_sum, current_mask)
        
    dfs2(0, 0, 0)
    
    # Reconstruct indices
    result_indices = []
    for i in range(len(idx1)):
        if best_mask1 & (1 << i):
            result_indices.append(idx1[i])
            
    for i in range(len(w2_order)):
        if best_mask2 & (1 << i):
            result_indices.append(idx2[i])
            
    if not result_indices:
        return [0]
        
    return result_indices
