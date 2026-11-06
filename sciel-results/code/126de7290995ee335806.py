import time
import numpy as np
from math import gcd
from functools import reduce

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit

    n = 30
    mid = n // 2
    
    # Weights for exact verification later (LCM of 1..30)
    def lcm(a, b):
        return a * b // gcd(a, b)
    L = reduce(lcm, range(1, n + 1))
    weights = [L // (i + 1) for i in range(n)]

    # Generate all signed sums for a set of indices using floats
    # indices: list of 0-based indices
    def generate_sums_float(indices):
        n_w = len(indices)
        # Start with 0
        sums = np.array([0.0], dtype=np.float64)
        codes = np.array([0], dtype=np.int64)
        
        for i, idx in enumerate(indices):
            w = 1.0 / (idx + 1)
            # Expand sums
            new_sums = np.empty(len(sums) * 3, dtype=np.float64)
            new_codes = np.empty(len(codes) * 3, dtype=np.int64)
            
            pow3_i = 3 ** i
            
            # Option 1: coeff = -1 (digit 0)
            new_sums[0::3] = sums - w
            new_codes[0::3] = codes
            
            # Option 2: coeff = 0 (digit 1)
            new_sums[1::3] = sums
            new_codes[1::3] = codes + pow3_i
            
            # Option 3: coeff = 1 (digit 2)
            new_sums[2::3] = sums + w
            new_codes[2::3] = codes + 2 * pow3_i
            
            sums = new_sums
            codes = new_codes
            
            if time.time() > deadline - 15:
                return None, None
                
        return sums, codes

    indices1 = list(range(mid))
    indices2 = list(range(mid, n))
    
    sums1, codes1 = generate_sums_float(indices1)
    if sums1 is None:
        return [1, -1] + [0] * 28

    sums2, codes2 = generate_sums_float(indices2)
    if sums2 is None:
        return [1, -1] + [0] * 28

    # Sort sums2
    sort_idx = np.argsort(sums2)
    sums2_sorted = sums2[sort_idx]
    codes2_sorted = codes2[sort_idx]
    
    N2 = len(sums2_sorted)
    
    # We want to minimize |s1 + s2| != 0
    # For each s1, s2 should be close to -s1
    targets = -sums1
    idx = np.searchsorted(sums2_sorted, targets)
    
    best_abs = float('inf')
    best_code1 = 0
    best_code2 = 0
    
    # Check idx, idx-1, idx+1
    # We will gather candidates to avoid multiple passes if possible, 
    # but vectorized check is fast enough.
    
    # Helper to update best
    def check_candidates(idx_c, s2_c):
        nonlocal best_abs, best_code1, best_code2
        sum_c = sums1 + s2_c
        abs_c = np.abs(sum_c)
        
        # Mask out zeros (or very small values that might be zero)
        # Threshold: 1/L is min non-zero. 1/L ~ 4e-13. 
        # Let's use 1e-14 as a safe "zero" threshold for float comparison.
        mask = abs_c > 1e-14
        
        if not np.any(mask):
            return
            
        abs_c_masked = np.where(mask, abs_c, float('inf'))
        min_c = np.min(abs_c_masked)
        
        if min_c < best_abs:
            idx_best = np.argmin(abs_c_masked)
            best_abs = min_c
            best_code1 = codes1[idx_best]
            best_code2 = codes2_sorted[idx_c[idx_best]]

    # Vectorized checks
    # 1. idx
    idx_c1 = np.clip(idx, 0, N2 - 1)
    s2_c1 = sums2_sorted[idx_c1]
    check_candidates(idx_c1, s2_c1)
    
    # 2. idx - 1
    idx_c0 = np.clip(idx - 1, 0, N2 - 1)
    s2_c0 = sums2_sorted[idx_c0]
    check_candidates(idx_c0, s2_c0)
    
    # 3. idx + 1
    idx_c2 = np.clip(idx + 1, 0, N2 - 1)
    s2_c2 = sums2_sorted[idx_c2]
    check_candidates(idx_c2, s2_c2)

    if best_abs == float('inf'):
        return [1, -1] + [0] * 28

    # Decode codes to coefficients
    def decode_code(code, n_coeffs, indices):
        coeffs = [0] * n_coeffs
        for i in range(n_coeffs):
            digit = code % 3
            # 0 -> -1, 1 -> 0, 2 -> 1
            if digit == 0:
                coeffs[i] = -1
            elif digit == 1:
                coeffs[i] = 0
            else:
                coeffs[i] = 1
            code //= 3
        return coeffs

    c1 = decode_code(best_code1, mid, indices1)
    c2 = decode_code(best_code2, n - mid, indices2)
    
    full_coeffs = c1 + c2
    
    # Final Verification: Ensure the sum is not zero.
    # Compute exact sum using integers
    exact_sum = 0
    for i in range(n):
        if full_coeffs[i] != 0:
            exact_sum += full_coeffs[i] * weights[i]
            
    if exact_sum == 0:
        # This should be rare, but if it happens, we need the next best.
        # For robustness, we can just return a known non-zero solution if this happens.
        # Or we can implement a loop to find the next best candidate.
        # Given the float gap, this is extremely unlikely.
        # Fallback to a simple non-zero solution
        return [1, -1] + [0] * 28
        
    return full_coeffs
