"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from math import gcd
from functools import reduce

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5 second safety margin
    
    # 1. Calculate LCM of 1..30
    def lcm(a, b):
        return abs(a * b) // gcd(a, b)
    
    L = reduce(lcm, range(1, 31))
    
    # 2. Calculate weights W_i = L // i
    weights = [L // i for i in range(1, 31)]
    
    # 3. Split weights into two halves
    half1_weights = weights[:15]
    half2_weights = weights[15:]
    
    def generate_sums_with_indices(ws):
        """
        Generate all possible signed sums for the given weights.
        Returns (sums_array, indices_array) where indices_array[i] is the base-3 index
        that produced sums_array[i].
        """
        sums = np.array([0], dtype=np.int64)
        indices = np.array([0], dtype=np.int64)
        for w in ws:
            # For each existing entry, create 3 new entries: -1, 0, +1
            n = len(sums)
            new_sums = np.empty(3 * n, dtype=np.int64)
            new_indices = np.empty(3 * n, dtype=np.int64)
            
            # coeff = -1: sums - w, index * 3 + 0
            new_sums[0:n] = sums - w
            new_indices[0:n] = indices * 3 + 0
            
            # coeff = 0: sums, index * 3 + 1
            new_sums[n:2*n] = sums
            new_indices[n:2*n] = indices * 3 + 1
            
            # coeff = +1: sums + w, index * 3 + 2
            new_sums[2*n:3*n] = sums + w
            new_indices[2*n:3*n] = indices * 3 + 2
            
            sums = new_sums
            indices = new_indices
        return sums, indices
    
    # Generate sums and indices for both halves
    sums1, idx1 = generate_sums_with_indices(half1_weights)
    sums2, idx2 = generate_sums_with_indices(half2_weights)
    
    # Check time
    if time.time() > deadline - 5:
        # Fallback
        return [0] * 28 + [1, -1]
    
    # 4. Sort sums2 and keep track of original indices
    sort_order = np.argsort(sums2)
    sorted_sums2 = sums2[sort_order]
    sorted_idx2 = idx2[sort_order]
    
    # 5. Find minimum non-zero sum
    idx = np.searchsorted(sorted_sums2, -sums1, side='left')
    
    idx_clipped = np.clip(idx, 0, len(sorted_sums2) - 1)
    idx_minus_1_clipped = np.clip(idx - 1, 0, len(sorted_sums2) - 1)
    
    cand_s2_1 = sorted_sums2[idx_clipped]
    cand_s2_2 = sorted_sums2[idx_minus_1_clipped]
    
    total_1 = sums1 + cand_s2_1
    total_2 = sums1 + cand_s2_2
    
    mask_1 = total_1 != 0
    mask_2 = total_2 != 0
    
    abs_total_1 = np.where(mask_1, np.abs(total_1), np.inf)
    abs_total_2 = np.where(mask_2, np.abs(total_2), np.inf)
    
    min_val_1 = np.min(abs_total_1)
    min_val_2 = np.min(abs_total_2)
    
    if min_val_1 <= min_val_2:
        if min_val_1 == np.inf:
            return [0] * 28 + [1, -1]
        
        k1_idx = np.argmin(abs_total_1)
        s1_val = sums1[k1_idx]
        s2_val = total_1[k1_idx] - s1_val
        
        # Get the index in the sorted array
        pos_in_sorted = idx_clipped[k1_idx]
        # Map back to original index in sums2
        orig_pos = sort_order[pos_in_sorted]
        idx_in_sums2 = orig_pos
        
        idx_in_sums1 = k1_idx
    else:
        if min_val_2 == np.inf:
            return [0] * 28 + [1, -1]
            
        k1_idx = np.argmin(abs_total_2)
        s1_val = sums1[k1_idx]
        s2_val = total_2[k1_idx] - s1_val
        
        pos_in_sorted = idx_minus_1_clipped[k1_idx]
        orig_pos = sort_order[pos_in_sorted]
        idx_in_sums2 = orig_pos
        
        idx_in_sums1 = k1_idx

    # 6. Decode indices to coefficients
    def decode_index(k, n_weights):
        """
        Decode a base-3 index back to coefficients.
        The generation process was:
        s = [0]
        for w in weights:
            s = [x-w, x, x+w for x in s]
        
        digit 0 -> -1
        digit 1 -> 0
        digit 2 -> +1
        
        The most significant digit corresponds to the last weight processed.
        """
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

    coeffs1 = decode_index(idx1[idx_in_sums1], len(half1_weights))
    coeffs2 = decode_index(idx2[idx_in_sums2], len(half2_weights))
    
    result = coeffs1 + coeffs2
    
    # Verify: compute the actual sum
    actual_sum = 0
    for i, c in enumerate(result):
        if c != 0:
            actual_sum += c * (L // (i + 1))
    
    if actual_sum == 0:
        # Should not happen, but fallback
        return [0] * 28 + [1, -1]
    
    return result

# EVOLVE-BLOCK-END
