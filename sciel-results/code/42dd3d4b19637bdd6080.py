import time
import numpy as np
from math import gcd
from functools import reduce

def construct():
    start_time = time.time()
    deadline = start_time + 80

    n = 30
    
    # Calculate LCM of 1..30
    def lcm(a, b):
        return abs(a * b) // gcd(a, b)
    
    L = reduce(lcm, range(1, n + 1))
    
    # Calculate A_i = L // i
    A = [L // i for i in range(1, n + 1)]
    
    # Split into two halves
    mid = n // 2
    A1 = A[:mid]
    A2 = A[mid:]
    
    # Generate sums for first half
    # Using list for generation, then convert to NumPy
    sums1 = [0]
    for a in A1:
        # Optimization: use list concatenation
        # This is O(3^k) total
        sums1 = sums1 + [x + a for x in sums1] + [x - a for x in sums1]
        
    if time.time() > deadline - 20:
        # Fallback to seed if generation is too slow
        return [0]*28 + [1, -1]

    # Generate sums for second half
    sums2 = [0]
    for a in A2:
        sums2 = sums2 + [x + a for x in sums2] + [x - a for x in sums2]
        
    if time.time() > deadline - 10:
        return [0]*28 + [1, -1]

    # Convert to NumPy arrays
    S1 = np.array(sums1, dtype=np.int64)
    S2 = np.array(sums2, dtype=np.int64)
    
    # Sort S1
    S1.sort()
    
    # Find best matches
    # We want s1 + s2 to be close to 0
    targets = -S2
    indices = np.searchsorted(S1, targets)
    
    # Check neighbors
    # Clip indices to valid range
    right_idx = np.clip(indices, 0, len(S1) - 1)
    left_idx = np.clip(indices - 1, 0, len(S1) - 1)
    
    right_vals = S1[right_idx]
    left_vals = S1[left_idx]
    
    diff_right = np.abs(right_vals + S2)
    diff_left = np.abs(left_vals + S2)
    
    # Determine which is smaller
    mask_right = diff_right < diff_left
    best_diffs = np.where(mask_right, diff_right, diff_left)
    
    # Exclude zero
    best_diffs[best_diffs == 0] = np.iinfo(np.int64).max
    
    min_diff = np.min(best_diffs)
    
    if min_diff == np.iinfo(np.int64).max:
        # All zero? Should not happen
        return [0]*28 + [1, -1]

    idx_min = np.argmin(best_diffs)
    
    s2_val = S2[idx_min]
    if mask_right[idx_min]:
        s1_val = right_vals[idx_min]
    else:
        s1_val = left_vals[idx_min]
        
    K_raw = s1_val + s2_val
    
    # Recover coefficients
    # K_30 = (30! / L) * K_raw
    fact = [1] * (n + 1)
    for i in range(1, n + 1):
        fact[i] = fact[i-1] * i
        
    M = fact[n] // L
    K = M * K_raw
    
    coeffs = [0] * n
    
    # Recover from n down to 1
    for i in range(n, 0, -1):
        found = False
        for c in (-1, 0, 1):
            # Check if (K - c * fact[i-1]) is divisible by i
            if (K - c * fact[i-1]) % i == 0:
                coeffs[i-1] = c
                K = (K - c * fact[i-1]) // i
                found = True
                break
        if not found:
            # Error in recovery, fallback
            return [0]*28 + [1, -1]
            
    # Verify sum is nonzero
    # The search ensured K_raw != 0, so sum != 0
    
    return coeffs

# EVOLVE-BLOCK-END
