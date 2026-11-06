"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import bisect
from math import gcd
from functools import reduce

def construct():
    t0 = time.time()
    deadline = t0 + 55.0  # Safety margin
    
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Compute LCM of 2..48
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = reduce(lcm, denominators)
    
    # Weights scaled by LCM
    weights = [L // d for d in denominators]
    
    # Split indices into two halves
    # n = 47. Split into 23 and 24.
    n_a = 23
    n_b = n - n_a
    
    # Indices for A and B
    idx_a = list(range(n_a))
    idx_b = list(range(n_a, n))
    
    w_a = [weights[i] for i in idx_a]
    w_b = [weights[i] for i in idx_b]
    
    # Generate subset sums for B
    # Size 2^24
    size_b = 1 << n_b
    sums_b = [0] * size_b
    # sums_b[0] = 0
    for i in range(1, size_b):
        lsb = i & -i
        j = lsb.bit_length() - 1
        sums_b[i] = sums_b[i ^ lsb] + w_b[j]
        
    # Pack (sum, mask) into single integer
    # sum < L, mask < 2^24
    # packed = (sum << 24) | mask
    # We only care about sum < L
    B_packed = []
    mask_limit = 1 << n_b
    for i in range(size_b):
        s = sums_b[i]
        if s < L:
            B_packed.append((s << n_b) | i)
            
    B_packed.sort()
    
    # Generate subset sums for A
    # Size 2^23
    size_a = 1 << n_a
    sums_a = [0] * size_a
    for i in range(1, size_a):
        lsb = i & -i
        j = lsb.bit_length() - 1
        sums_a[i] = sums_a[i ^ lsb] + w_a[j]
        
    best_sum = -1
    best_mask_a = 0
    best_mask_b = 0
    
    max_mask_b = mask_limit - 1
    
    for i in range(size_a):
        s_a = sums_a[i]
        if s_a >= L:
            continue
            
        # We need s_b such that s_a + s_b < L  => s_b <= L - 1 - s_a
        target = L - 1 - s_a
        if target < 0:
            continue
            
        # Binary search in B_packed for largest element <= (target << n_b) | max_mask_b
        # Since B_packed is sorted by value, and value is (sum << n_b) | mask,
        # all entries with sum <= target will have value <= (target << n_b) | max_mask_b
        # provided we handle the mask part correctly.
        # Actually, if sum < target, any mask is fine.
        # If sum == target, we want the largest mask? No, we just want any valid mask, 
        # but we want to maximize sum. The sum is the primary key.
        # So we just want the last element in B_packed that is <= (target << n_b) | max_mask_b.
        
        search_key = (target << n_b) | max_mask_b
        idx = bisect.bisect_right(B_packed, search_key) - 1
        
        if idx >= 0:
            p = B_packed[idx]
            s_b = p >> n_b
            mask_b = p & max_mask_b
            
            # Verify constraint (should be true by construction, but safe)
            if s_a + s_b < L:
                total = s_a + s_b
                if total > best_sum:
                    best_sum = total
                    best_mask_a = i
                    best_mask_b = mask_b
                    
    # Construct final indices
    final_indices = []
    for j in range(n_a):
        if best_mask_a & (1 << j):
            final_indices.append(idx_a[j])
            
    for j in range(n_b):
        if best_mask_b & (1 << j):
            final_indices.append(idx_b[j])
        
    return sorted(final_indices)
# EVOLVE-BLOCK-END
