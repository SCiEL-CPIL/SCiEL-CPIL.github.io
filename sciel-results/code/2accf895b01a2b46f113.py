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
    
    def generate_sums(weights_half):
        sums = np.array([0], dtype=np.int64)
        for w in weights_half:
            sums = np.concatenate([sums - w, sums, sums + w])
        return sums

    sums1 = generate_sums(w1)
    sums2 = generate_sums(w2)
    
    indices1 = np.arange(len(sums1), dtype=np.int64)
    sorter = np.argsort(sums1)
    sums1_sorted = sums1[sorter]
    sorted_indices1 = indices1[sorter]
    
    len_s1 = len(sums1_sorted)
    
    targets = -sums2
    
    idx_l = np.searchsorted(sums1_sorted, targets, side='left')
    idx_r = np.searchsorted(sums1_sorted, targets, side='right')
    
    idx_cand1 = idx_l - 1
    valid1 = idx_cand1 >= 0
    idx_cand1_clipped = np.clip(idx_cand1, 0, len_s1 - 1)
    s1_cand1 = sums1_sorted[idx_cand1_clipped]
    
    total1 = sums2 + s1_cand1
    abs_total1 = np.abs(total1)
    
    mask1 = valid1 & (total1 != 0)
    abs_total1[~mask1] = np.iinfo(np.int64).max
    
    idx_cand2 = idx_r
    valid2 = idx_cand2 < len_s1
    idx_cand2_clipped = np.clip(idx_cand2, 0, len_s1 - 1)
    s1_cand2 = sums1_sorted[idx_cand2_clipped]
    
    total2 = sums2 + s1_cand2
    abs_total2 = np.abs(total2)
    
    mask2 = valid2 & (total2 != 0)
    abs_total2[~mask2] = np.iinfo(np.int64).max
    
    min_abs1 = np.min(abs_total1)
    min_abs2 = np.min(abs_total2)
    
    if min_abs1 < min_abs2:
        best_abs = min_abs1
        best_j = np.argmin(abs_total1)
        best_idx1_sorted = idx_cand1_clipped[best_j]
        best_idx2_orig = best_j
    else:
        best_abs = min_abs2
        best_j = np.argmin(abs_total2)
        best_idx1_sorted = idx_cand2_clipped[best_j]
        best_idx2_orig = best_j
        
    best_idx1_orig = sorted_indices1[best_idx1_sorted]
    
    powers_of_3 = [1] * mid
    for i in range(1, mid):
        powers_of_3[i] = powers_of_3[i-1] * 3
        
    def reconstruct_from_index(index, n_w, powers):
        coeffs = [0] * n_w
        for j in range(n_w):
            digit = (index // powers[j]) % 3
            coeffs[j] = digit - 1
        return coeffs

    c1 = reconstruct_from_index(best_idx1_orig, mid, powers_of_3)
    
    powers_of_3_2 = [1] * (n - mid)
    for i in range(1, n - mid):
        powers_of_3_2[i] = powers_of_3_2[i-1] * 3
        
    c2 = reconstruct_from_index(best_idx2_orig, n - mid, powers_of_3_2)
    
    final_coeffs = c1 + c2
    
    val_num = 0
    for i in range(n):
        val_num += final_coeffs[i] * (L // (i + 1))
        
    if val_num == 0:
        final_coeffs = [1] + [0] * (n - 1)
        
    return final_coeffs
