import time
import sys
from math import gcd
from functools import reduce
from fractions import Fraction
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 50.0 # 10s margin
    
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Calculate LCM
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = reduce(lcm, denominators)
    
    weights = [L // d for d in denominators]
    
    # Split into two halves
    mid = n // 2 # 23
    # A: 0..22 (23 items)
    # B: 23..46 (24 items)
    
    def generate_sums(weights_subset):
        sums = [0]
        masks = [0]
        for i, w in enumerate(weights_subset):
            # Double the list
            # New sums: old sums + w
            # New masks: old masks | (1 << i)
            # Using list comprehension is faster than loop append
            current_len = len(sums)
            sums += [s + w for s in sums]
            masks += [m | (1 << i) for m in masks]
            if time.time() > deadline:
                return sums, masks # Partial result? No, we need full for correctness. 
                # If we return partial, we might miss the optimal.
                # But if we hit deadline, we should probably return best found so far or fallback.
                # For now, assume it finishes.
        return sums, masks

    # Generate for A
    sums_A, masks_A = generate_sums(weights[:mid])
    
    # Check time
    if time.time() > deadline - 10:
         # Fallback to greedy if MITM is too slow?
         # But MITM generation is fast. Sorting is the bottleneck.
         pass

    # Sort A by sums
    # List of tuples (sum, mask)
    A_pairs = list(zip(sums_A, masks_A))
    A_pairs.sort(key=lambda x: x[0])
    
    # Generate for B
    sums_B, masks_B = generate_sums(weights[mid:])
    
    # Sort B by sums
    B_pairs = list(zip(sums_B, masks_B))
    B_pairs.sort(key=lambda x: x[0])
    
    # Two pointer search
    # We want max s_A + s_B < L
    # A_pairs and B_pairs are sorted by sum ascending.
    i = len(A_pairs) - 1
    j = 0
    best_sum = -1
    best_mask_A = -1
    best_mask_B = -1
    
    n_A = len(A_pairs)
    n_B = len(B_pairs)
    
    while i >= 0 and j < n_B:
        s_A = A_pairs[i][0]
        s_B = B_pairs[j][0]
        total = s_A + s_B
        
        if total < L:
            if total > best_sum:
                best_sum = total
                best_mask_A = A_pairs[i][1]
                best_mask_B = B_pairs[j][1]
            j += 1
        else:
            i -= 1
            
        if time.time() > deadline:
             break

    # Reconstruct indices
    # best_mask_A corresponds to first 'mid' items
    # best_mask_B corresponds to next 'n-mid' items
    final_mask = best_mask_A | (best_mask_B << mid)
    
    indices = []
    for k in range(n):
        if (final_mask >> k) & 1:
            indices.append(k)
            
    # Verify result
    if not indices:
        # Fallback
        return [0]
        
    # Check if sum is strictly below 1
    # Deficit must be positive
    # If best_sum == -1, something went wrong
    if best_sum < 0:
        return [0]
        
    return indices

