import time
import numpy as np
from math import gcd
from functools import reduce

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin for 90s limit

    n = 30
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    # Calculate LCM of 1..30
    L = reduce(lcm, range(1, n + 1))
    
    # Weights: L // i for i in 1..30
    weights = np.array([L // (i + 1) for i in range(n)], dtype=np.int64)
    
    mid = n // 2
    w1 = weights[:mid]
    w2 = weights[mid:]
    
    def generate_sums_and_codes(weights_part):
        n_w = len(weights_part)
        sums = np.array([0], dtype=np.int64)
        codes = np.array([0], dtype=np.int64)
        
        for i, w in enumerate(weights_part):
            new_sums = np.empty(len(sums) * 3, dtype=np.int64)
            new_codes = np.empty(len(codes) * 3, dtype=np.int64)
            
            pow3_i = 3 ** i
            
            new_sums[0::3] = sums - w
            new_codes[0::3] = codes
            
            new_sums[1::3] = sums
            new_codes[1::3] = codes + pow3_i
            
            new_sums[2::3] = sums + w
            new_codes[2::3] = codes + 2 * pow3_i
            
            sums = new_sums
            codes = new_codes
            
            if time.time() > deadline - 15:
                return None, None
                
        return sums, codes

    sums1, codes1 = generate_sums_and_codes(w1)
    if sums1 is None:
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
    
    targets = -sums1
    idx = np.searchsorted(sums2_sorted, targets)
    
    best_abs = INF
    best_code1 = 0
    best_code2 = 0
    
    # Check a window of candidates around each target
    # Use a wider window to find smaller |sum|
    window = 5  # Check idx-5 to idx+5
    
    for offset in range(-window, window + 1):
        idx_c = np.clip(idx + offset, 0, N2 - 1)
        s2_c = sums2_sorted[idx_c]
        sum_c = sums1 + s2_c
        abs_c = np.abs(sum_c)
        abs_c[sum_c == 0] = INF
        
        min_c = np.min(abs_c)
        if min_c < best_abs:
            best_abs = min_c
            idx_best = np.argmin(abs_c)
            best_code1 = codes1[idx_best]
            best_code2 = codes2_sorted[idx_c[idx_best]]
            
            if best_abs == 1:
                break  # Can't do better than 1
    
    if best_abs == INF:
        return [1, -1] + [0] * 28

    def decode_code(code, n_coeffs):
        coeffs = []
        for i in range(n_coeffs):
            digit = code % 3
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
    coeffs = c1 + c2
    
    # Local search refinement: try flipping individual coefficients
    # and see if we can reduce |sum| further
    current_sum = best_abs  # This is |sum(c_i * w_i)|, the scaled sum
    current_sign = 1  # We'll track the actual signed sum
    
    # Recompute the actual signed sum
    actual_sum = sum(coeffs[i] * weights[i] for i in range(n))
    current_abs = abs(actual_sum)
    
    # Try single-flip local search
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for i in range(n):
            # Try changing coeffs[i] to -1, 0, 1 (if different)
            old_val = coeffs[i]
            for new_val in [-1, 0, 1]:
                if new_val == old_val:
                    continue
                # Compute new sum
                delta = (new_val - old_val) * weights[i]
                new_sum = actual_sum + delta
                new_abs = abs(new_sum)
                
                if new_abs < current_abs and new_abs != 0:
                    coeffs[i] = new_val
                    actual_sum = new_sum
                    current_abs = new_abs
                    improved = True
                    break
    
    return coeffs
