import time
import random
import numpy as np
from math import gcd
from functools import reduce

def construct():
    start_time = time.time()
    time_limit = 85.0  # Safety margin for 90s limit

    def lcm(a, b):
        return abs(a * b) // gcd(a, b)
    
    # LCM of 1..30
    L = reduce(lcm, range(1, 31))
    
    # Precompute weights A[i] = L // (i+1) for i in 0..29
    weights = np.array([L // (i + 1) for i in range(30)], dtype=np.int64)
    
    half = 15
    w1 = weights[:half]
    w2 = weights[half:]
    
    # Precompute powers of 3
    p3 = [1] * (half + 1)
    for i in range(1, half + 1):
        p3[i] = p3[i-1] * 3
        
    n_combos = p3[half]
    
    # Generate indices
    idxs = np.arange(n_combos, dtype=np.int64)
    
    sums1 = np.zeros(n_combos, dtype=np.int64)
    sums2 = np.zeros(n_combos, dtype=np.int64)
    
    for i in range(half):
        p = p3[i]
        # c = (idxs // p) % 3 - 1
        c = (idxs // p) % 3 - 1
        
        sums1 += c * w1[i]
        sums2 += c * w2[i]
        
    if time.time() - start_time > time_limit * 0.5:
        # If we are running out of time, we can fall back to a simpler heuristic
        # but MITM is fast.
        pass

    # Sort sums2 and keep track of original indices
    sorted_idx2 = np.argsort(sums2)
    sorted_sums2 = sums2[sorted_idx2]
    
    # Find best match for each sum in sums1
    targets = -sums1
    pos = np.searchsorted(sorted_sums2, targets, side='left')
    
    # Handle boundaries
    pos_clamped = np.clip(pos, 0, n_combos - 1)
    pos_minus_1 = np.clip(pos - 1, 0, n_combos - 1)
    
    s2_pos = sorted_sums2[pos_clamped]
    s2_minus = sorted_sums2[pos_minus_1]
    
    total_pos = sums1 + s2_pos
    total_minus = sums1 + s2_minus
    
    # Filter out zeros
    mask_pos = total_pos != 0
    mask_minus = total_minus != 0
    
    abs_pos = np.abs(total_pos)
    abs_minus = np.abs(total_minus)
    
    # Set invalid ones to large value
    max_int = np.iinfo(np.int64).max
    abs_pos[~mask_pos] = max_int
    abs_minus[~mask_minus] = max_int
    
    # Find global min
    min_pos_val = np.min(abs_pos)
    min_minus_val = np.min(abs_minus)
    
    if min_pos_val < min_minus_val:
        best_val = min_pos_val
        idx1_best = np.argmin(abs_pos)
        idx2_sorted_best = pos_clamped[idx1_best]
    else:
        best_val = min_minus_val
        idx1_best = np.argmin(abs_minus)
        idx2_sorted_best = pos_minus_1[idx1_best]
        
    # Get original indices
    best_idx1 = idx1_best
    best_idx2 = sorted_idx2[idx2_sorted_best]
    
    # Reconstruct coefficients
    coeffs = [0] * 30
    for i in range(half):
        coeffs[i] = (best_idx1 // p3[i]) % 3 - 1
        coeffs[half + i] = (best_idx2 // p3[i]) % 3 - 1
        
    return coeffs
