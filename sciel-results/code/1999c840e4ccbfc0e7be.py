"""Combined program: Meet-in-the-Middle exact search for optimal subset sum."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from functools import reduce
from bisect import bisect_left

def construct():
    start_time = time.time()
    timeout = 55.0  # Safety margin for 60s limit (evaluator kills at 90s, but let's be safe)
    
    # Denominators 2 to 48 inclusive
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Compute LCM of 2..48
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = reduce(lcm, range(2, 49))
    
    # Precompute integer weights
    # index i corresponds to denominator i+2
    # weight[i] = L // (i+2)
    weights = [L // d for d in denominators]
    
    # Split into two halves
    mid = n // 2
    n1 = mid
    n2 = n - mid
    
    w1 = weights[:mid]
    w2 = weights[mid:]
    
    # Generate subset sums for first half
    # We need sum and mask for each subset
    # To save memory and time, we can generate lists
    size1 = 1 << n1
    size2 = 1 << n2
    
    # Optimization: Generate sums using iterative doubling
    sums1 = [0] * size1
    masks1 = [0] * size1
    for i in range(n1):
        w = w1[i]
        for j in range(1 << i):
            sums1[(1 << i) | j] = sums1[j] + w
            masks1[(1 << i) | j] = masks1[j] | (1 << i)
            
    # We don't actually need masks1 for the search, only for reconstruction at the end
    # But we need to know which mask1 gave the best sum.
    
    # Generate subset sums for second half
    # We will store (sum, mask) pairs, filter, sort
    sums2 = [0] * size2
    masks2 = [0] * size2
    for i in range(n2):
        w = w2[i]
        for j in range(1 << i):
            sums2[(1 << i) | j] = sums2[j] + w
            masks2[(1 << i) | j] = masks2[j] | (1 << i)
            
    # Prepare second half for binary search
    # We need a list of (sum, mask) where sum < L
    # Sort by sum
    # To save memory, we can filter first
    valid_pairs2 = []
    for mask in range(size2):
        s = sums2[mask]
        if s < L:
            valid_pairs2.append((s, mask))
            
    # Sort by sum
    valid_pairs2.sort(key=lambda x: x[0])
    
    # Extract sums for bisect
    sorted_sums2 = [p[0] for p in valid_pairs2]
    
    best_sum = -1
    best_mask1 = 0
    best_mask2 = 0
    
    # Iterate over first half
    for mask1 in range(size1):
        s1 = sums1[mask1]
        if s1 >= L:
            continue
            
        # We want s2 such that s1 + s2 < L
        # s2 < L - s1
        target = L - s1
        
        # bisect_left finds the first index where sorted_sums2[idx] >= target
        # We want the largest s2 < target, so index - 1
        idx = bisect_left(sorted_sums2, target)
        
        if idx > 0:
            # The best candidate in second half is at idx - 1
            s2, m2 = valid_pairs2[idx - 1]
            total_sum = s1 + s2
            
            # We strictly require total_sum < L.
            # Since s1 < L and s2 < L - s1, total_sum < L is guaranteed.
            
            if total_sum > best_sum:
                best_sum = total_sum
                best_mask1 = mask1
                best_mask2 = m2
                
    # Check time
    if time.time() - start_time > timeout:
         # If we timed out during search (unlikely for MITM), we can't easily return partial
         # But MITM is fast. If we are here, we are done.
         pass

    # Reconstruct indices
    indices = []
    for i in range(n1):
        if best_mask1 & (1 << i):
            indices.append(i)
            
    for i in range(n2):
        if best_mask2 & (1 << i):
            indices.append(mid + i)
            
    # Ensure non-empty
    # The empty set has sum 0.
    # Is it possible the best sum is 0?
    # Only if no non-empty subset has sum < L.
    # But 1/2 = L/2 < L. So index 0 (denom 2) is a valid non-empty subset.
    # So best_sum will be at least L/2.
    # Thus indices will not be empty.
    
    if not indices:
        # Fallback just in case
        indices = [0]
        
    return indices
# EVOLVE-BLOCK-END
