import numpy as np
import time
from math import gcd
from fractions import Fraction
from bisect import bisect_left

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s limit

    n = 30
    half = 15
    
    # 1. Calculate LCM of 1..30
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = 1
    for i in range(1, n + 1):
        L = lcm(L, i)
    
    # Precompute scaled terms: term[i] = L / (i+1) for i in 0..29
    terms = np.array([L // (i + 1) for i in range(n)], dtype=np.int64)
    
    # Split into two halves
    terms_A = terms[:half]
    terms_B = terms[half:]
    
    size = 3 ** half  # 14,348,907
    
    # 2. Generate all sums for first half using numpy
    # Use the recursive generation approach with numpy
    
    def gen_sums_numpy(terms_part):
        """Generate all sums for a set of terms using numpy."""
        current = np.array([0], dtype=np.int64)
        
        for t in terms_part:
            # new_sums = [s - t, s, s + t for s in current]
            new_sums = np.concatenate([current - t, current, current + t])
            current = new_sums
            
        return current
    
    # Generate sums for A
    sums_A = gen_sums_numpy(terms_A)
    
    if time.time() > deadline:
        return [0]*28 + [1, -1]
    
    # Sort sums_A
    sort_idx_A = np.argsort(sums_A)
    sums_A_sorted = sums_A[sort_idx_A]
    
    # Generate sums for B
    sums_B = gen_sums_numpy(terms_B)
    
    if time.time() > deadline:
        return [0]*28 + [1, -1]
    
    # 3. Search for minimum nonzero absolute sum
    # For each sum in B, find closest in A to -sum_B
    # Use vectorized search
    
    targets = -sums_B
    
    # searchsorted returns the index where target would be inserted
    pos = np.searchsorted(sums_A_sorted, targets)
    
    # Clip to valid range
    pos_clipped_0 = np.clip(pos, 0, len(sums_A_sorted) - 1)
    pos_clipped_1 = np.clip(pos - 1, 0, len(sums_A_sorted) - 1)
    
    # Get candidate sums from A
    cand_A_0 = sums_A_sorted[pos_clipped_0]
    cand_A_1 = sums_A_sorted[pos_clipped_1]
    
    # Calculate total sums
    total_0 = cand_A_0 + sums_B
    total_1 = cand_A_1 + sums_B
    
    # Find minimum nonzero absolute value
    abs_total_0 = np.abs(total_0)
    abs_total_1 = np.abs(total_1)
    
    # Mask zeros
    mask_0 = total_0 != 0
    mask_1 = total_1 != 0
    
    # Set zeros to a large value
    max_int = np.iinfo(np.int64).max
    abs_total_0[~mask_0] = max_int
    abs_total_1[~mask_1] = max_int
    
    # Find global minimum
    min_val_0 = np.min(abs_total_0)
    min_idx_0 = np.argmin(abs_total_0)
    
    min_val_1 = np.min(abs_total_1)
    min_idx_1 = np.argmin(abs_total_1)
    
    if min_val_0 < min_val_1:
        best_abs_val = min_val_0
        best_pos_A = pos_clipped_0[min_idx_0]
        best_idx_B = min_idx_0
    else:
        best_abs_val = min_val_1
        best_pos_A = pos_clipped_1[min_idx_1]
        best_idx_B = min_idx_1
    
    if best_abs_val is None or best_abs_val == max_int:
        # Fallback
        return [0]*28 + [1, -1]
        
    # 4. Recover coefficients
    # We need to find which base-3 index produced sums_A_sorted[best_pos_A]
    # and which base-3 index produced sums_B[best_idx_B]
    
    # For A: we need to find the index in the original (unsorted) sums_A that maps to best_pos_A in sorted
    # sums_A_sorted[best_pos_A] = sums_A[sort_idx_A[best_pos_A]]
    # So the original index is sort_idx_A[best_pos_A]
    orig_idx_A = sort_idx_A[best_pos_A]
    
    # For B: best_idx_B is already the index in the original sums_B
    orig_idx_B = best_idx_B
    
    # Now decode base-3 indices to get coefficients
    coeffs = np.zeros(30, dtype=int)
    
    powers = 3 ** np.arange(half)
    
    # For A
    digits_A = (orig_idx_A // powers) % 3
    coeffs[0:half] = np.where(digits_A == 0, -1, np.where(digits_A == 1, 0, 1))
    
    # For B
    digits_B = (orig_idx_B // powers) % 3
    coeffs[half:30] = np.where(digits_B == 0, -1, np.where(digits_B == 1, 0, 1))
    
    # 5. Verify
    total_frac = Fraction(0)
    for i in range(n):
        if coeffs[i] != 0:
            total_frac += Fraction(int(coeffs[i]), i + 1)
    
    if total_frac == 0:
        # Fallback
        return [0]*28 + [1, -1]
    
    return [int(x) for x in coeffs]
