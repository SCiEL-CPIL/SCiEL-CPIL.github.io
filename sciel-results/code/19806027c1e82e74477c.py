import time
import numpy as np
from math import gcd
from functools import reduce

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit

    n = 30
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    # Calculate LCM of 1..30
    L = reduce(lcm, range(1, n + 1))
    
    # Weights: L // i for i in 1..30
    # The sum is sum(c_i / i) = (1/L) * sum(c_i * (L/i))
    # So we minimize |sum(c_i * w_i)| where w_i = L // i
    weights = np.array([L // (i + 1) for i in range(n)], dtype=np.int64)
    
    mid = n // 2
    w1 = weights[:mid]
    w2 = weights[mid:]
    
    # Function to generate all sums and their base-3 codes
    # Code is an integer where digit i (base 3) represents coeff[i]
    # coeff -1 -> 0, 0 -> 1, 1 -> 2
    # This allows easy decoding later.
    def generate_sums_and_codes(weights_part):
        n_w = len(weights_part)
        # Start with sum=0, code=0
        sums = np.array([0], dtype=np.int64)
        codes = np.array([0], dtype=np.int64)
        
        for i, w in enumerate(weights_part):
            # For each existing sum, create 3 new ones: -w, 0, +w
            # New sums:
            new_sums = np.empty(len(sums) * 3, dtype=np.int64)
            new_codes = np.empty(len(codes) * 3, dtype=np.int64)
            
            # Option 1: coeff = -1 (digit 0)
            new_sums[0::3] = sums - w
            new_codes[0::3] = codes  # digit i becomes 0, so code doesn't change at pos i? 
            # Wait, base-3 encoding:
            # If we build codes by appending digits, we need to shift existing digits.
            # Let's define code such that the least significant digit is the first coefficient.
            # If we process weight i, we are setting the i-th digit.
            # Existing codes have digits 0..i-1.
            # New code = old_code * 3^i + digit? No, that's if we append to the right.
            # Let's use the standard: code = sum(c_j * 3^j).
            # When we add a new weight w_i at index i, we are setting digit i.
            # The previous digits 0..i-1 are unchanged.
            # So new_code = old_code + digit * (3^i).
            # This is efficient.
            
            pow3_i = 3 ** i
            
            # Option 1: coeff = -1 (digit 0)
            # new_code = old_code + 0
            new_sums[0::3] = sums - w
            new_codes[0::3] = codes
            
            # Option 2: coeff = 0 (digit 1)
            # new_code = old_code + 1 * 3^i
            new_sums[1::3] = sums
            new_codes[1::3] = codes + pow3_i
            
            # Option 3: coeff = 1 (digit 2)
            # new_code = old_code + 2 * 3^i
            new_sums[2::3] = sums + w
            new_codes[2::3] = codes + 2 * pow3_i
            
            sums = new_sums
            codes = new_codes
            
            if time.time() > deadline - 10:
                # Fallback if generation is too slow (unlikely for N=15)
                return None, None
                
        return sums, codes

    sums1, codes1 = generate_sums_and_codes(w1)
    if sums1 is None:
        # Emergency fallback: [1, -1] + [0]*28 -> sum = 1 - 1/2 = 1/2. Value 0.5.
        # Or just return something valid.
        return [1, -1] + [0] * 28

    sums2, codes2 = generate_sums_and_codes(w2)
    if sums2 is None:
        return [1, -1] + [0] * 28

    # Sort sums2 and keep codes aligned
    sort_idx = np.argsort(sums2)
    sums2_sorted = sums2[sort_idx]
    codes2_sorted = codes2[sort_idx]
    
    N2 = len(sums2_sorted)
    INF = np.iinfo(np.int64).max
    
    # We want to minimize |s1 + s2| != 0
    # For each s1, we want s2 close to -s1.
    
    targets = -sums1
    idx = np.searchsorted(sums2_sorted, targets)
    
    # Candidates: idx-1, idx, idx+1
    # We will check these 3 positions for each s1.
    
    best_abs = INF
    best_code1 = 0
    best_code2 = 0
    
    # Vectorized check for idx
    idx_c1 = np.clip(idx, 0, N2 - 1)
    s2_c1 = sums2_sorted[idx_c1]
    sum_c1 = sums1 + s2_c1
    abs_c1 = np.abs(sum_c1)
    abs_c1[sum_c1 == 0] = INF
    
    min_c1 = np.min(abs_c1)
    if min_c1 < best_abs:
        best_abs = min_c1
        idx_best = np.argmin(abs_c1)
        best_code1 = codes1[idx_best]
        best_code2 = codes2_sorted[idx_c1[idx_best]]
        
    # Vectorized check for idx-1
    idx_c0 = np.clip(idx - 1, 0, N2 - 1)
    s2_c0 = sums2_sorted[idx_c0]
    sum_c0 = sums1 + s2_c0
    abs_c0 = np.abs(sum_c0)
    abs_c0[sum_c0 == 0] = INF
    
    min_c0 = np.min(abs_c0)
    if min_c0 < best_abs:
        best_abs = min_c0
        idx_best = np.argmin(abs_c0)
        best_code1 = codes1[idx_best]
        best_code2 = codes2_sorted[idx_c0[idx_best]]
        
    # Vectorized check for idx+1
    idx_c2 = np.clip(idx + 1, 0, N2 - 1)
    s2_c2 = sums2_sorted[idx_c2]
    sum_c2 = sums1 + s2_c2
    abs_c2 = np.abs(sum_c2)
    abs_c2[sum_c2 == 0] = INF
    
    min_c2 = np.min(abs_c2)
    if min_c2 < best_abs:
        best_abs = min_c2
        idx_best = np.argmin(abs_c2)
        best_code1 = codes1[idx_best]
        best_code2 = codes2_sorted[idx_c2[idx_best]]

    if best_abs == INF:
        # Should not happen, but fallback
        return [1, -1] + [0] * 28

    # Decode codes to coefficients
    def decode_code(code, n_coeffs):
        coeffs = []
        for i in range(n_coeffs):
            digit = code % 3
            # 0 -> -1, 1 -> 0, 2 -> 1
            if digit == 0:
                coeffs.append(-1)
            elif digit == 1:
                coeffs.append(0)
            else:
                coeffs.append(1)
            code //= 3
        return coeffs

    c1 = decode_code(best_code1, mid)
    c2 = decode_code(best_code2, n - mid)
    
    return c1 + c2
