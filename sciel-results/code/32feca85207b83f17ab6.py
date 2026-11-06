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
    
    # weights[i] corresponds to coefficient for 1/(i+1), so weight = L // (i+1)
    # Sum of c[i] * weights[i] / L = sum c[i]/(i+1)
    weights = np.array([L // (i + 1) for i in range(n)], dtype=np.int64)
    
    # Meet-in-the-middle: split into two halves
    mid = n // 2  # 15
    w1 = weights[:mid]
    w2 = weights[mid:]
    
    def generate_sums(weights_half):
        """
        Generate all possible sums for a half using coefficients in {-1, 0, 1}.
        Returns a numpy array of sums.
        The order of sums corresponds to base-3 representation of the index,
        where the least significant digit corresponds to the first weight,
        and the most significant digit corresponds to the last weight.
        """
        sums = np.array([0], dtype=np.int64)
        for w in weights_half:
            sums = np.concatenate([sums - w, sums, sums + w])
        return sums
    
    # Generate sums for both halves
    sums1 = generate_sums(w1)
    sums2 = generate_sums(w2)
    
    # Sort sums2 and keep track of original indices for reconstruction
    sorted_sums2, sorted_indices2 = np.sort(sums2), np.argsort(sums2)
    
    best_abs_S = np.iinfo(np.int64).max
    best_idx1 = -1
    best_idx2_sorted = -1
    
    # Vectorized search
    targets = -sums1
    indices = np.searchsorted(sorted_sums2, targets)
    
    len_s2 = len(sorted_sums2)
    
    # Check candidates: idx-1, idx, idx+1
    # Right candidates (first element >= target)
    idx_r = np.clip(indices, 0, len_s2 - 1)
    s2_r = sorted_sums2[idx_r]
    total_r = sums1 + s2_r
    
    # Left candidates (last element < target)
    idx_l = np.clip(indices - 1, 0, len_s2 - 1)
    s2_l = sorted_sums2[idx_l]
    total_l = sums1 + s2_l
    
    # Also check idx+1 for completeness
    idx_rp1 = np.clip(indices + 1, 0, len_s2 - 1)
    s2_rp1 = sorted_sums2[idx_rp1]
    total_rp1 = sums1 + s2_rp1
    
    # We want min |total| for total != 0
    abs_total_r = np.abs(total_r)
    abs_total_l = np.abs(total_l)
    abs_total_rp1 = np.abs(total_rp1)
    
    # Mask out zeros
    INF = np.iinfo(np.int64).max
    abs_total_r[total_r == 0] = INF
    abs_total_l[total_l == 0] = INF
    abs_total_rp1[total_rp1 == 0] = INF
    
    # Find the global minimum among the three candidate sets
    min_abs_r = np.min(abs_total_r)
    min_abs_l = np.min(abs_total_l)
    min_abs_rp1 = np.min(abs_total_rp1)
    
    best_min = min(min_abs_r, min_abs_l, min_abs_rp1)
    
    if best_min < min_abs_r:
        if best_min < min_abs_rp1:
            best_abs_S = best_min
            best_idx1 = np.argmin(abs_total_l)
            best_idx2_sorted = idx_l[best_idx1]
        else:
            best_abs_S = best_min
            best_idx1 = np.argmin(abs_total_rp1)
            best_idx2_sorted = idx_rp1[best_idx1]
    elif best_min < min_abs_rp1:
        best_abs_S = best_min
        best_idx1 = np.argmin(abs_total_r)
        best_idx2_sorted = idx_r[best_idx1]
    else:
        best_abs_S = best_min
        best_idx1 = np.argmin(abs_total_rp1)
        best_idx2_sorted = idx_rp1[best_idx1]
        
    # Check if we found a valid solution
    if best_abs_S == INF:
        # This implies all candidates were zero, which is unlikely.
        # Fallback: scan neighbors of the first s1
        s1_val = sums1[0]
        target = -s1_val
        pos = np.searchsorted(sorted_sums2, target)
        for delta in range(0, 100):
            for d in [delta, -delta]:
                if delta == 0 and d < 0: continue
                idx = pos + d
                if 0 <= idx < len_s2:
                    s2_val = sorted_sums2[idx]
                    total = s1_val + s2_val
                    if total != 0:
                        best_abs_S = abs(total)
                        best_idx1 = 0
                        best_idx2_sorted = idx
                        break
            if best_abs_S != INF:
                break
    
    if best_abs_S == INF:
        return [0] * 28 + [1, -1]
    
    # Reconstruct coefficients
    orig_idx1 = best_idx1
    orig_idx2 = sorted_indices2[best_idx2_sorted]
    
    def decode_coeffs(idx, n_weights):
        """
        Decode base-3 index to coefficients.
        Least significant digit corresponds to weight[0].
        """
        coeffs = np.zeros(n_weights, dtype=np.int64)
        for i in range(n_weights):
            digit = idx % 3
            coeffs[i] = digit - 1
            idx //= 3
        return coeffs
    
    c1 = decode_coeffs(orig_idx1, mid)
    c2 = decode_coeffs(orig_idx2, n - mid)
    
    final_coeffs = np.concatenate([c1, c2])
    
    # Verify the result
    val_num = sum(int(c[i]) * int(weights[i]) for i in range(n))
    if val_num == 0:
        # Should not happen, but fallback
        return [0] * 28 + [1, -1]
    
    # Ensure we return Python ints
    return [int(c) for c in final_coeffs]
