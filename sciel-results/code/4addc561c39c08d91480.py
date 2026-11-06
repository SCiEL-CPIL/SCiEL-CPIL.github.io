import math
import numpy as np
import time
from functools import reduce

def construct():
    start_time = time.time()
    deadline = start_time + 50
    
    n = 30
    H = reduce(math.lcm, range(1, n + 1))
    W = [H // i for i in range(1, n + 1)]
    
    mid = 15
    W_a = W[:mid]
    W_b = W[mid:]
    
    # Generate all sums and codes for A
    # Code: base-3 representation of coefficients (-1, 0, 1) mapped to (0, 1, 2)
    # c = -1 -> 0, c = 0 -> 1, c = 1 -> 2
    # code = sum (c_i + 1) * 3^i
    
    def generate_sums(weights):
        sums = [0]
        codes = [0]
        power = 1
        for w in weights:
            new_sums = []
            new_codes = []
            # Unroll for speed? Or just list comprehension
            for s, c in zip(sums, codes):
                # c_new = -1 (digit 0)
                new_sums.append(s - w)
                new_codes.append(c)
                # c_new = 0 (digit 1)
                new_sums.append(s)
                new_codes.append(c + power)
                # c_new = 1 (digit 2)
                new_sums.append(s + w)
                new_codes.append(c + 2 * power)
            sums = new_sums
            codes = new_codes
            power *= 3
        return sums, codes

    if time.time() > deadline:
        return _fallback(W, n)

    sums_a, codes_a = generate_sums(W_a)
    
    if time.time() > deadline:
        return _fallback(W, n)

    # Sort A by sum
    # We need to sort pairs (sum, code)
    # Using numpy for sorting is faster
    arr_sums_a = np.array(sums_a, dtype=np.int64)
    arr_codes_a = np.array(codes_a, dtype=np.int32)
    
    # Get indices that would sort arr_sums_a
    sort_idx = np.argsort(arr_sums_a, kind='mergesort')
    arr_sums_a_sorted = arr_sums_a[sort_idx]
    arr_codes_a_sorted = arr_codes_a[sort_idx]
    
    if time.time() > deadline:
        return _fallback(W, n)

    # Generate B
    sums_b, codes_b = generate_sums(W_b)
    arr_sums_b = np.array(sums_b, dtype=np.int64)
    arr_codes_b = np.array(codes_b, dtype=np.int32)
    
    if time.time() > deadline:
        return _fallback(W, n)

    # Find best match
    # For each b, we want a such that a + b is non-zero and minimal |a+b|
    # a approx -b
    
    targets = -arr_sums_b
    # searchsorted returns index where target would be inserted
    # idx is the index of the first element >= target
    idxs = np.searchsorted(arr_sums_a_sorted, targets)
    
    # We need to check idx and idx-1
    # Handle boundaries
    len_a = len(arr_sums_a_sorted)
    
    # Initialize best value and indices
    best_val = np.inf
    best_idx_a = -1
    best_idx_b = -1
    
    # Candidate 1: idx
    idxs_c1 = np.clip(idxs, 0, len_a - 1)
    sums_a_c1 = arr_sums_a_sorted[idxs_c1]
    total_c1 = sums_a_c1 + arr_sums_b
    
    # Candidate 2: idx - 1
    idxs_c2 = np.clip(idxs - 1, 0, len_a - 1)
    sums_a_c2 = arr_sums_a_sorted[idxs_c2]
    total_c2 = sums_a_c2 + arr_sums_b
    
    # Combine candidates
    # We need to find the min abs(total) where total != 0
    # Mask out zeros
    mask_c1 = total_c1 != 0
    mask_c2 = total_c2 != 0
    
    # Replace zeros with inf to ignore them in min calculation
    abs_c1 = np.abs(total_c1)
    abs_c1[~mask_c1] = np.inf
    
    abs_c2 = np.abs(total_c2)
    abs_c2[~mask_c2] = np.inf
    
    # Find global min
    # We have two arrays of abs values corresponding to the same b index
    # min_abs = min(abs_c1, abs_c2)
    # But we need to know which one (c1 or c2) was the min to get the correct code
    
    # Let's flatten and find the single best
    # Create a combined array of (value, source_idx, type)
    # Or just loop over the smaller set of candidates?
    # No, vectorize the comparison.
    
    # min_idx in the combined space
    # Let's just compare element-wise
    # is_c1_better = abs_c1 < abs_c2
    # But if one is inf, the other wins.
    
    # Construct final best candidate for each b
    best_abs_per_b = np.minimum(abs_c1, abs_c2)
    
    # Find the b index with the smallest best_abs
    # If all are inf, no solution found (should not happen)
    min_b_idx = np.argmin(best_abs_per_b)
    min_val = best_abs_per_b[min_b_idx]
    
    if np.isinf(min_val):
        return _fallback(W, n)
        
    # Determine if c1 or c2 was the winner for this b
    if abs_c1[min_b_idx] <= abs_c2[min_b_idx]:
        # Winner is c1
        idx_a = idxs_c1[min_b_idx]
        idx_b = min_b_idx
        code_a = arr_codes_a_sorted[idx_a]
        code_b = arr_codes_b[idx_b]
    else:
        # Winner is c2
        idx_a = idxs_c2[min_b_idx]
        idx_b = min_b_idx
        code_a = arr_codes_a_sorted[idx_a]
        code_b = arr_codes_b[idx_b]
        
    # Decode codes to coefficients
    coeffs_a = decode_code(code_a, mid)
    coeffs_b = decode_code(code_b, n - mid)
    
    final_coeffs = coeffs_a + coeffs_b
    
    # Verify
    total = sum(c * W[i] for i, c in enumerate(final_coeffs))
    if total == 0:
        # This shouldn't happen due to masks, but just in case
        return _fallback(W, n)
        
    return final_coeffs

def decode_code(code, length):
    coeffs = [0] * length
    for i in range(length):
        digit = code % 3
        code //= 3
        # digit 0 -> -1, 1 -> 0, 2 -> 1
        coeffs[i] = digit - 1
    return coeffs

def _fallback(W, n):
    # Simple greedy fallback
    coeffs = [0] * n
    curr_sum = 0
    for i in range(n, 0, -1):
        w = W[i - 1]
        best_c = 0
        best_val_local = curr_sum
        for c in [-1, 0, 1]:
            val = curr_sum + c * w
            if abs(val) < abs(best_val_local):
                best_val_local = val
                best_c = c
        coeffs[i - 1] = best_c
        curr_sum = best_val_local
        
    total = sum(c * W[i] for i, c in enumerate(coeffs))
    if total == 0:
        min_idx = min(range(n), key=lambda i: W[i])
        if coeffs[min_idx] == 1:
            coeffs[min_idx] = -1
        else:
            coeffs[min_idx] = 1
    return coeffs
