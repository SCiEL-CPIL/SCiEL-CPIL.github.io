"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import lcm
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin from 60s limit
    
    denominators = list(range(2, 49))
    n = len(denominators)  # 47 items
    
    # Compute LCM of all denominators
    L = 1
    for d in denominators:
        L = lcm(L, d)
    
    # Weights: W_i = L / d_i. We want to maximize sum(W_i) subject to sum(W_i) < L.
    weights = [L // d for d in denominators]
    
    # Meet-in-the-middle approach
    # Split into two halves
    n1 = n // 2  # 23
    n2 = n - n1  # 24
    
    left_weights = weights[:n1]
    right_weights = weights[n1:]
    
    # Generate all subset sums for left half (2^23 = 8,388,608 entries)
    left_size = 1 << n1
    left_sums = [0] * left_size
    for i in range(n1):
        bit = 1 << i
        w = left_weights[i]
        for mask in range(bit):
            left_sums[bit | mask] = left_sums[mask] + w
    
    # Generate all subset sums for right half (2^24 = 16,777,216 entries)
    right_size = 1 << n2
    right_sums = [0] * right_size
    for i in range(n2):
        bit = 1 << i
        w = right_weights[i]
        for mask in range(bit):
            right_sums[bit | mask] = right_sums[mask] + w
    
    # Sort right sums with their masks for binary search
    right_pairs = [(right_sums[mask], mask) for mask in range(right_size)]
    right_pairs.sort()
    right_sum_values = [p[0] for p in right_pairs]
    
    # Find best combination
    best_sum = -1
    best_left_mask = 0
    best_right_mask = 0
    
    for mask in range(left_size):
        ls = left_sums[mask]
        if ls >= L:
            continue
        # We need rs such that ls + rs < L, i.e., rs < L - ls
        # Find the largest rs < L - ls
        limit = L - ls
        idx = bisect.bisect_left(right_sum_values, limit) - 1
        if idx >= 0:
            rs = right_sum_values[idx]
            total = ls + rs
            if total > best_sum:
                best_sum = total
                best_left_mask = mask
                best_right_mask = right_pairs[idx][1]
    
    # Check time
    if time.time() > deadline:
        # Fallback: return best found so far or greedy
        pass
    
    # Reconstruct indices
    result_indices = []
    for i in range(n1):
        if best_left_mask & (1 << i):
            result_indices.append(i)
    for i in range(n2):
        if best_right_mask & (1 << i):
            result_indices.append(n1 + i)
    
    # Ensure non-empty
    if not result_indices:
        result_indices = [0]
    
    return result_indices
# EVOLVE-BLOCK-END
