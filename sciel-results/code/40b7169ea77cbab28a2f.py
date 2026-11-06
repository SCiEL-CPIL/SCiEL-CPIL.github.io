import time
import numpy as np
from math import gcd
from functools import reduce

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    n = 30
    # Calculate LCM of 1..30
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = reduce(lcm, range(1, n + 1))
    
    # Precompute weights
    # w[i] = L / (i+1) for i in 0..29
    weights = np.array([L // (i + 1) for i in range(n)], dtype=np.int64)
    
    # Split into two halves
    mid = n // 2
    w1 = weights[:mid]
    w2 = weights[mid:]
    
    def generate_sums_with_codes(weights_half):
        """
        Generate all possible sums and their encoded coefficient patterns for a half.
        Returns (sums_array, codes_array) where codes are base-3 encoded integers.
        """
        n_half = len(weights_half)
        sums = np.array([0], dtype=np.int64)
        codes = np.array([0], dtype=np.int64)
        
        for i, w in enumerate(weights_half):
            # For each existing sum, create three new sums: s-w, s, s+w
            # We want to append -1, 0, 1 for the i-th coefficient
            # Code update: new_code = old_code * 3 + digit
            # Digits: -1 -> 0, 0 -> 1, 1 -> 2
            
            n_existing = len(sums)
            
            # Repeat existing sums and codes 3 times
            # Order: (all old with -1), (all old with 0), (all old with 1)
            sums_new = np.concatenate([sums, sums, sums])
            codes_new = np.concatenate([codes, codes, codes])
            
            # Update codes
            # First third: digit 0 (represents -1)
            codes_new[:n_existing] = codes_new[:n_existing] * 3 + 0
            # Second third: digit 1 (represents 0)
            codes_new[n_existing:2*n_existing] = codes_new[n_existing:2*n_existing] * 3 + 1
            # Third third: digit 2 (represents 1)
            codes_new[2*n_existing:] = codes_new[2*n_existing:] * 3 + 2
            
            # Update sums
            sums_new[:n_existing] = sums_new[:n_existing] - w
            sums_new[n_existing:2*n_existing] = sums_new[n_existing:2*n_existing] # already 0
            sums_new[2*n_existing:] = sums_new[2*n_existing:] + w
            
            sums = sums_new
            codes = codes_new
        
        return sums, codes

    # Generate sums and codes for both halves
    sums1, codes1 = generate_sums_with_codes(w1)
    sums2, codes2 = generate_sums_with_codes(w2)
    
    # Sort sums2 and keep track of original indices
    sort_idx = np.argsort(sums2)
    sums2_sorted = sums2[sort_idx]
    codes2_sorted = codes2[sort_idx]
    
    len_s2 = len(sums2_sorted)
    
    # We want to find pairs (s1, s2) such that |s1 + s2| is minimized and non-zero.
    # Vectorized search
    targets = -sums1
    
    # Find insertion points
    indices = np.searchsorted(sums2_sorted, targets)
    
    # Check multiple neighbors: indices-2 to indices+2
    # Create candidate index arrays
    candidates = []
    for offset in [-2, -1, 0, 1, 2]:
        idx = np.clip(indices + offset, 0, len_s2 - 1)
        s2_vals = sums2_sorted[idx]
        total_vals = sums1 + s2_vals
        candidates.append(total_vals)
    
    # Stack all candidates
    all_totals = np.vstack(candidates)  # shape (5, len_sums1)
    
    # Mask zeros
    mask = all_totals != 0
    
    # Set zeros to max value
    abs_totals = np.abs(all_totals).astype(np.int64)
    abs_totals[~mask] = np.iinfo(np.int64).max
    
    # Find global minimum
    flat_idx = np.argmin(abs_totals)
    
    # Recover which candidate and which sums1 index
    n_sums1 = len(sums1)
    row = flat_idx // n_sums1
    col = flat_idx % n_sums1
    
    s1_val = sums1[col]
    code1 = codes1[col]
    
    # Get the corresponding s2 index
    offset = [-2, -1, 0, 1, 2][row]
    idx = np.clip(indices[col] + offset, 0, len_s2 - 1)
    s2_val = sums2_sorted[idx]
    code2 = codes2_sorted[idx]
    
    # Decode codes to coefficients
    def decode_code(code, n_coeffs):
        coeffs = []
        for _ in range(n_coeffs):
            digit = code % 3
            # Map 0->-1, 1->0, 2->1
            coeffs.append(digit - 1)
            code //= 3
        return coeffs
    
    result1 = decode_code(int(code1), mid)
    result2 = decode_code(int(code2), n - mid)
    
    return result1 + result2
