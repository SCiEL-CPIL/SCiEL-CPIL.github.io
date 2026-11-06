import time
from math import gcd
from fractions import Fraction
import sys
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 80.0

    denominators = list(range(2, 49))
    n = len(denominators)

    # Calculate LCM
    L = 1
    for d in denominators:
        L = L * d // gcd(L, d)

    weights = [L // d for d in denominators]

    def check_time():
        return time.time() > deadline

    def gen_subsets_fast(items):
        """Generates (sum, mask) for all subsets of items where sum < L."""
        pairs = [(0, 0)]
        for i in range(len(items)):
            w = weights[items[i]]
            bit = 1 << i
            current_len = len(pairs)
            # Iterate backwards to avoid modifying list while iterating? 
            # No, we are appending. We need to iterate over the *original* part.
            # So we take a snapshot of the length.
            for j in range(current_len):
                s, m = pairs[j]
                ns = s + w
                if ns < L:
                    pairs.append((ns, m | bit))
        return pairs

    # Split items
    # n = 47
    # half1: 0..22 (23 items)
    # half2: 23..46 (24 items)
    
    half1_indices = list(range(23))
    half2_indices = list(range(23, 47))
    
    n1 = len(half1_indices)
    n2 = len(half2_indices)
    
    if check_time():
        return [0]

    # Generate subsets for half 1
    pairs1 = gen_subsets_fast(half1_indices)
    
    if check_time():
        # Fallback
        return [0]

    # Sort pairs1 by sum
    pairs1.sort(key=lambda x: x[0])
    
    # Extract sums for binary search
    sums1 = [p[0] for p in pairs1]
    
    # Generate subsets for half 2
    pairs2 = gen_subsets_fast(half2_indices)
    
    if check_time():
        # Fallback: use best from half 1
        if pairs1:
            best_s, best_m = pairs1[-1]
            indices = []
            for i in range(n1):
                if best_m & (1 << i):
                    indices.append(half1_indices[i])
            return indices
        return [0]

    best_total_sum = -1
    best_mask1 = 0
    best_mask2 = 0
    
    # Iterate over half 2 subsets
    for s2, m2 in pairs2:
        if check_time():
            break
            
        limit = L - s2
        if limit <= 0:
            continue
            
        # Find rightmost index in sums1 where value < limit
        # bisect_left(sums1, limit) gives first index where value >= limit
        # We want the one before that.
        pos = bisect.bisect_left(sums1, limit)
        
        if pos > 0:
            idx1 = pos - 1
            s1 = sums1[idx1]
            total = s1 + s2
            if total > best_total_sum:
                best_total_sum = total
                best_mask1 = pairs1[idx1][1]
                best_mask2 = m2
                
    # Reconstruct indices
    indices = []
    for i in range(n1):
        if best_mask1 & (1 << i):
            indices.append(half1_indices[i])
    for i in range(n2):
        if best_mask2 & (1 << i):
            indices.append(half2_indices[i])
    
    # Ensure non-empty
    if not indices:
        # Find any valid non-empty subset
        # e.g. index 0 (denom 2, sum 0.5 < 1)
        indices = [0]
        
    return indices
