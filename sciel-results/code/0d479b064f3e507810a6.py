import math
import numpy as np
from functools import reduce

def construct():
    # Calculate LCM of 1 to 30
    L = reduce(math.lcm, range(1, 31))
    
    # Weights
    weights = [L // i for i in range(1, 31)]
    
    # Split weights
    w1 = weights[:15]
    w2 = weights[15:]
    
    def gen_np(w):
        sums = np.array([0], dtype=np.int64)
        codes = np.array([0], dtype=np.int64)
        for i, wt in enumerate(w):
            bit1 = 1 << (2*i)
            bit2 = 2 << (2*i)
            
            # Expand arrays
            sums_new = np.concatenate([sums, sums + wt, sums - wt])
            codes_new = np.concatenate([codes, codes + bit1, codes + bit2])
            
            sums = sums_new
            codes = codes_new
        return sums, codes

    s1_sums, s1_codes = gen_np(w1)
    s2_sums, s2_codes = gen_np(w2)
    
    # Sort s2_sums and reorder s2_codes
    sort_idx = np.argsort(s2_sums)
    s2_sums = s2_sums[sort_idx]
    s2_codes = s2_codes[sort_idx]
    
    # For each s1, find best s2 such that s1 + s2 != 0 and |s1+s2| is min
    # Use searchsorted to find insertion point for -s1
    # idx is the first index such that s2_sums[idx] >= -s1
    # Candidates are idx and idx-1
    
    targets = -s1_sums
    idxs = np.searchsorted(s2_sums, targets)
    
    # Handle boundaries
    # idxs can be 0 or len(s2_sums)
    
    # Candidate 1: idxs
    # Candidate 2: idxs - 1
    
    # We need to be careful with indices
    valid1 = (idxs < len(s2_sums)) & (idxs >= 0)
    valid2 = (idxs - 1 < len(s2_sums)) & (idxs - 1 >= 0)
    
    # Initialize best values
    best_abs = np.full(len(s1_sums), np.iinfo(np.int64).max, dtype=np.int64)
    best_code1 = s1_codes
    best_code2 = np.zeros(len(s1_sums), dtype=np.int64)
    best_sum = np.zeros(len(s1_sums), dtype=np.int64)
    
    # Process valid1
    if np.any(valid1):
        idxs_v1 = idxs[valid1]
        s1_v1 = s1_sums[valid1]
        s2_v1 = s2_sums[idxs_v1]
        sum_v1 = s1_v1 + s2_v1
        
        # Exclude 0
        non_zero_v1 = sum_v1 != 0
        if np.any(non_zero_v1):
            idxs_v1_nz = idxs_v1[non_zero_v1]
            s1_v1_nz = s1_v1[non_zero_v1]
            sum_v1_nz = sum_v1[non_zero_v1]
            abs_v1 = np.abs(sum_v1_nz)
            
            # Update best if this is better
            # We need to map back to original indices
            # This is tricky with boolean masks.
            # Let's do it in a loop or use advanced indexing carefully.
            # Given the size, a loop over 14M items in Python is too slow.
            # We need vectorized update.
            
            # Create arrays for all valid1 indices
            # Actually, let's just compute the best for each s1 independently using vectorized ops on the whole array?
            # No, the indices are different.
            
            # Alternative:
            # Compute sum_v1 for all valid1.
            # Compute sum_v2 for all valid2.
            # Combine them?
            
            pass

    # The vectorized update with different valid masks is complex.
    # Let's try a simpler approach:
    # Since we just need the global minimum, we can compute all candidate sums and find the min.
    
    # Candidate Set 1: (s1, s2[idx])
    # Candidate Set 2: (s1, s2[idx-1])
    
    # Let's gather all valid candidates into two arrays.
    
    # Set 1
    idx1 = idxs
    valid1 = (idx1 < len(s2_sums)) & (idx1 >= 0)
    s1_1 = s1_sums[valid1]
    s2_1 = s2_sums[idx1[valid1]]
    sum_1 = s1_1 + s2_1
    code1_1 = s1_codes[valid1]
    code2_1 = s2_codes[idx1[valid1]]
    
    # Set 2
    idx2 = idxs - 1
    valid2 = (idx2 < len(s2_sums)) & (idx2 >= 0)
    s1_2 = s1_sums[valid2]
    s2_2 = s2_sums[idx2[valid2]]
    sum_2 = s1_2 + s2_2
    code1_2 = s1_codes[valid2]
    code2_2 = s2_codes[idx2[valid2]]
    
    # Concatenate
    all_sums = np.concatenate([sum_1, sum_2])
    all_code1 = np.concatenate([code1_1, code1_2])
    all_code2 = np.concatenate([code2_1, code2_2])
    
    # Filter non-zero
    non_zero_mask = all_sums != 0
    all_sums = all_sums[non_zero_mask]
    all_code1 = all_code1[non_zero_mask]
    all_code2 = all_code2[non_zero_mask]
    
    # Find min absolute value
    if len(all_sums) == 0:
        # Fallback: return all zeros? But sum must be non-zero.
        # This shouldn't happen.
        return [0]*30
        
    abs_sums = np.abs(all_sums)
    min_idx = np.argmin(abs_sums)
    
    best_sum = all_sums[min_idx]
    best_code1 = all_code1[min_idx]
    best_code2 = all_code2[min_idx]
    
    # Reconstruct coefficients
    coeffs = [0] * 30
    for k in range(15):
        d1 = (best_code1 >> (2*k)) & 3
        d2 = (best_code2 >> (2*k)) & 3
        if d1 == 1: coeffs[k] = 1
        elif d1 == 2: coeffs[k] = -1
        else: coeffs[k] = 0
        
        if d2 == 1: coeffs[15+k] = 1
        elif d2 == 2: coeffs[15+k] = -1
        else: coeffs[15+k] = 0
        
    return coeffs
