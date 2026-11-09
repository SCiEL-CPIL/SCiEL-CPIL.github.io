import time
import bisect
from math import lcm

def construct():
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Calculate LCM
    L = 1
    for d in denominators:
        L = lcm(L, d)
        
    weights = [L // d for d in denominators]
    
    half = n // 2
    left_w = weights[:half]
    right_w = weights[half:]
    
    # Generate left sums
    # Using a list of length 2^len(left_w)
    size_l = 1 << len(left_w)
    left_sums = [0] * size_l
    for i, w in enumerate(left_w):
        bit = 1 << i
        for mask in range(bit):
            left_sums[bit | mask] = left_sums[mask] + w
            
    # Generate right sums
    size_r = 1 << len(right_w)
    right_sums = [0] * size_r
    for i, w in enumerate(right_w):
        bit = 1 << i
        for mask in range(bit):
            right_sums[bit | mask] = right_sums[mask] + w
            
    # Create list of (sum, mask) for right
    # This is the most memory intensive part
    right_pairs = list(zip(right_sums, range(size_r)))
    right_pairs.sort()
    right_vals = [p[0] for p in right_pairs]
    
    best_deficit = L
    best_l_mask = 0
    best_r_mask = 0
    
    # Iterate left
    for mask in range(size_l):
        ls = left_sums[mask]
        if ls >= L:
            continue
        limit = L - ls
        idx = bisect.bisect_left(right_vals, limit) - 1
        if idx >= 0:
            rs = right_vals[idx]
            total = ls + rs
            deficit = L - total
            if deficit < best_deficit:
                best_deficit = deficit
                best_l_mask = mask
                best_r_mask = right_pairs[idx][1]
                
    res = []
    for i in range(len(left_w)):
        if best_l_mask & (1 << i):
            res.append(i)
    for i in range(len(right_w)):
        if best_r_mask & (1 << i):
            res.append(half + i)
            
    return res
