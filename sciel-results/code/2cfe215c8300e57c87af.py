import time
import numpy as np
from math import gcd
from functools import reduce

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    
    # 1. Calculate LCM of 1..30
    def lcm(a, b):
        return abs(a * b) // gcd(a, b)
    
    L = reduce(lcm, range(1, 31))
    
    # 2. Calculate weights W_i = L // i
    weights = np.array([L // i for i in range(1, 31)], dtype=np.int64)
    
    # Split into two halves
    half1_weights = weights[:15]
    half2_weights = weights[15:]
    
    def generate_sums(ws):
        """
        Generate all possible signed sums for the given weights.
        Returns a NumPy array of sums.
        The order corresponds to base-3 representation where:
        digit 0 -> -1, digit 1 -> 0, digit 2 -> +1
        Most significant digit is the last weight processed.
        """
        sums = np.array([0], dtype=np.int64)
        for w in ws:
            n = len(sums)
            # Create new array of size 3*n
            # Coeff -1: sums - w
            # Coeff 0: sums
            # Coeff +1: sums + w
            new_sums = np.empty(3 * n, dtype=np.int64)
            new_sums[0:n] = sums - w
            new_sums[n:2*n] = sums
            new_sums[2*n:3*n] = sums + w
            sums = new_sums
        return sums

    # Generate sums for both halves
    sums1 = generate_sums(half1_weights)
    sums2 = generate_sums(half2_weights)
    
    if time.time() > deadline - 10:
        # Fallback if generation is unexpectedly slow (should not happen)
        return [0] * 28 + [1, -1]

    # Sort sums2 and keep original indices
    sort_order = np.argsort(sums2)
    sorted_sums2 = sums2[sort_order]
    
    # Vectorized search for closest -sums1 in sorted_sums2
    # target = -sums1
    # We want to find j such that |sums1[i] + sums2[j]| is minimized and non-zero.
    
    # searchsorted returns the index where target would be inserted to maintain order.
    # side='left' means if target exists, it returns the first occurrence.
    pos = np.searchsorted(sorted_sums2, -sums1, side='left')
    
    # Candidate indices in sorted_sums2
    # We check pos and pos-1
    # Handle boundaries: pos can be 0 or len(sorted_sums2)
    
    n2 = len(sorted_sums2)
    
    # Candidate 1: pos
    pos1 = np.clip(pos, 0, n2 - 1)
    cand_s2_1 = sorted_sums2[pos1]
    total_1 = sums1 + cand_s2_1
    
    # Candidate 2: pos - 1
    pos2 = np.clip(pos - 1, 0, n2 - 1)
    cand_s2_2 = sorted_sums2[pos2]
    total_2 = sums1 + cand_s2_2
    
    # Mask out zeros
    mask_1 = total_1 != 0
    mask_2 = total_2 != 0
    
    # Set invalid entries to infinity for min calculation
    abs_total_1 = np.where(mask_1, np.abs(total_1), np.int64(np.iinfo(np.int64).max))
    abs_total_2 = np.where(mask_2, np.abs(total_2), np.int64(np.iinfo(np.int64).max))
    
    # Find global minimum
    min_val_1 = np.min(abs_total_1)
    min_val_2 = np.min(abs_total_2)
    
    if min_val_1 <= min_val_2:
        if min_val_1 == np.iinfo(np.int64).max:
            return [0] * 28 + [1, -1]
        
        k1_idx = np.argmin(abs_total_1)
        # The index in the original sums2 array
        # pos1[k1_idx] is the index in sorted_sums2
        # sort_order maps sorted index to original index
        orig_idx_in_sums2 = sort_order[pos1[k1_idx]]
        idx_in_sums1 = k1_idx
    else:
        if min_val_2 == np.iinfo(np.int64).max:
            return [0] * 28 + [1, -1]
            
        k1_idx = np.argmin(abs_total_2)
        orig_idx_in_sums2 = sort_order[pos2[k1_idx]]
        idx_in_sums1 = k1_idx

    # Decode indices to coefficients
    def decode_index(k, n_weights):
        coeffs = []
        for _ in range(n_weights):
            digit = k % 3
            if digit == 0:
                coeffs.append(-1)
            elif digit == 1:
                coeffs.append(0)
            else:
                coeffs.append(1)
            k //= 3
        coeffs.reverse()
        return coeffs

    coeffs1 = decode_index(idx_in_sums1, len(half1_weights))
    coeffs2 = decode_index(orig_idx_in_sums2, len(half2_weights))
    
    result = coeffs1 + coeffs2
    
    # Verify
    actual_sum = 0
    for i, c in enumerate(result):
        if c != 0:
            actual_sum += c * (L // (i + 1))
            
    if actual_sum == 0:
        # This should not happen due to masking, but just in case
        return [0] * 28 + [1, -1]
        
    return result
