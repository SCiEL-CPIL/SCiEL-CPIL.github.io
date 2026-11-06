import time
import numpy as np
from math import gcd
from functools import reduce

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = 30
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = reduce(lcm, range(1, n + 1))
    
    weights = np.array([L // (i + 1) for i in range(n)], dtype=np.int64)
    
    mid = n // 2
    w1 = weights[:mid]
    w2 = weights[mid:]
    
    def generate_sums_with_codes(weights_half):
        # Returns array of sums and array of codes (base 3)
        sums = np.array([0], dtype=np.int64)
        codes = np.array([0], dtype=np.int64)
        for i, w in enumerate(weights_half):
            # New sums: s-w, s, s+w
            # New codes: c + 0*3^i, c + 1*3^i, c + 2*3^i
            # Note: coefficient -1 maps to 0, 0 maps to 1, 1 maps to 2 in base 3?
            # Let's map -1->0, 0->1, 1->2.
            # Then coeff = digit - 1.
            
            p = 3 ** i
            new_sums = np.concatenate([sums - w, sums, sums + w])
            new_codes = np.concatenate([codes, codes + p, codes + 2*p])
            sums = new_sums
            codes = new_codes
        return sums, codes

    sums1, codes1 = generate_sums_with_codes(w1)
    sums2, codes2 = generate_sums_with_codes(w2)
    
    sums2_sorted, idx_sort = np.sort(sums2), np.argsort(sums2)
    codes2_sorted = codes2[idx_sort]
    N2 = len(sums2_sorted)
    
    # Vectorized search
    targets = -sums1
    idx = np.searchsorted(sums2_sorted, targets)
    
    idx_c0 = np.clip(idx - 1, 0, N2 - 1)
    idx_c1 = np.clip(idx, 0, N2 - 1)
    idx_c2 = np.clip(idx + 1, 0, N2 - 1)
    
    k2_c0 = sums2_sorted[idx_c0]
    k2_c1 = sums2_sorted[idx_c1]
    k2_c2 = sums2_sorted[idx_c2]
    
    sum_c0 = sums1 + k2_c0
    sum_c1 = sums1 + k2_c1
    sum_c2 = sums1 + k2_c2
    
    abs_c0 = np.abs(sum_c0)
    abs_c1 = np.abs(sum_c1)
    abs_c2 = np.abs(sum_c2)
    
    INF = np.iinfo(np.int64).max
    abs_c0[sum_c0 == 0] = INF
    abs_c1[sum_c1 == 0] = INF
    abs_c2[sum_c2 == 0] = INF
    
    min0 = np.min(abs_c0)
    min1 = np.min(abs_c1)
    min2 = np.min(abs_c2)
    
    best_abs = min(min0, min1, min2)
    
    if best_abs == INF:
        return [0]*30

    if best_abs == min0:
        idx_best = np.argmin(abs_c0)
        c1_code = codes1[idx_best]
        c2_code = codes2_sorted[idx_c0[idx_best]]
    elif best_abs == min1:
        idx_best = np.argmin(abs_c1)
        c1_code = codes1[idx_best]
        c2_code = codes2_sorted[idx_c1[idx_best]]
    else:
        idx_best = np.argmin(abs_c2)
        c1_code = codes1[idx_best]
        c2_code = codes2_sorted[idx_c2[idx_best]]
        
    def decode_code(code, length):
        coeffs = []
        for i in range(length):
            digit = code % 3
            code //= 3
            # 0 -> -1, 1 -> 0, 2 -> 1
            coeffs.append(digit - 1)
        return coeffs

    c1 = decode_code(c1_code, mid)
    c2 = decode_code(c2_code, n - mid)
    
    return c1 + c2
