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
    
    def generate_sums(weights_half):
        """
        Generate all possible sums for a half using coefficients in {-1, 0, 1}.
        """
        sums = np.array([0], dtype=np.int64)
        for w in weights_half:
            # sums_new = [s-w for s in sums] + [s for s in sums] + [s+w for s in sums]
            sums = np.concatenate([sums - w, sums, sums + w])
        return sums

    # Generate sums for both halves
    # 3^15 is ~14.3 million
    sums1 = generate_sums(w1)
    sums2 = generate_sums(w2)
    
    # Sort sums2 for binary search
    sums2_sorted = np.sort(sums2)
    
    len_s2 = len(sums2_sorted)
    
    # We want to find pairs (s1, s2) such that |s1 + s2| is minimized and non-zero.
    # Vectorized search
    targets = -sums1
    
    # Find insertion points
    indices = np.searchsorted(sums2_sorted, targets)
    
    # We need to check neighbors: indices, indices-1, indices+1
    
    # Candidate 1: sums2_sorted[indices]
    idx_r = np.clip(indices, 0, len_s2 - 1)
    s2_r = sums2_sorted[idx_r]
    total_r = sums1 + s2_r
    abs_r = np.abs(total_r)
    # Mask zeros
    abs_r[total_r == 0] = np.iinfo(np.int64).max
    
    # Candidate 2: sums2_sorted[indices - 1]
    idx_l = np.clip(indices - 1, 0, len_s2 - 1)
    s2_l = sums2_sorted[idx_l]
    total_l = sums1 + s2_l
    abs_l = np.abs(total_l)
    abs_l[total_l == 0] = np.iinfo(np.int64).max
    
    # Candidate 3: sums2_sorted[indices + 1]
    idx_rr = np.clip(indices + 1, 0, len_s2 - 1)
    s2_rr = sums2_sorted[idx_rr]
    total_rr = sums1 + s2_rr
    abs_rr = np.abs(total_rr)
    abs_rr[total_rr == 0] = np.iinfo(np.int64).max
    
    # Find minimums for each candidate set
    min_r = np.min(abs_r)
    min_l = np.min(abs_l)
    min_rr = np.min(abs_rr)
    
    # Determine which candidate has the global minimum
    # If ties, pick any (they will have the same value)
    if min_r <= min_l and min_r <= min_rr:
        min_val = min_r
        idx_col = np.argmin(abs_r)
        s1_val = sums1[idx_col]
        s2_val = s2_r[idx_col]
    elif min_l <= min_r and min_l <= min_rr:
        min_val = min_l
        idx_col = np.argmin(abs_l)
        s1_val = sums1[idx_col]
        s2_val = s2_l[idx_col]
    else:
        min_val = min_rr
        idx_col = np.argmin(abs_rr)
        s1_val = sums1[idx_col]
        s2_val = s2_rr[idx_col]
        
    # Reconstruct coefficients
    # Precompute sum-to-coeffs maps for sub-halves to speed up reconstruction
    def build_sum_map(weights_sub):
        """
        Build a dictionary mapping sums to coefficients for a sub-half.
        """
        n_sub = len(weights_sub)
        sums_map = {0: []}
        for i, w in enumerate(weights_sub):
            w_int = int(w)
            new_sums_map = {}
            for s, coeffs in sums_map.items():
                # -w
                ns = s - w_int
                new_coeffs = coeffs + [-1]
                if ns not in new_sums_map:
                    new_sums_map[ns] = new_coeffs
                # 0
                ns = s
                new_coeffs = coeffs + [0]
                if ns not in new_sums_map:
                    new_sums_map[ns] = new_coeffs
                # +w
                ns = s + w_int
                new_coeffs = coeffs + [1]
                if ns not in new_sums_map:
                    new_sums_map[ns] = new_coeffs
            sums_map = new_sums_map
        return sums_map

    # Split w1 and w2 into sub-halves
    mid1 = len(w1) // 2
    mid2 = len(w2) // 2
    
    w1a = w1[:mid1]
    w1b = w1[mid1:]
    w2a = w2[:mid2]
    w2b = w2[mid2:]
    
    map_w1a = build_sum_map(w1a)
    map_w1b = build_sum_map(w1b)
    map_w2a = build_sum_map(w2a)
    map_w2b = build_sum_map(w2b)
    
    def find_coeffs_from_maps(map_a, map_b, target_sum):
        """
        Find coefficients given precomputed maps for two sub-halves.
        """
        target_sum = int(target_sum)
        # Iterate over the smaller map for efficiency
        if len(map_a) < len(map_b):
            for s_a, coeffs_a in map_a.items():
                target_b = target_sum - s_a
                if target_b in map_b:
                    coeffs_b = map_b[target_b]
                    return coeffs_a + coeffs_b
        else:
            for s_b, coeffs_b in map_b.items():
                target_a = target_sum - s_b
                if target_a in map_a:
                    coeffs_a = map_a[target_a]
                    return coeffs_a + coeffs_b
        return None

    coeffs1 = find_coeffs_from_maps(map_w1a, map_w1b, s1_val)
    coeffs2 = find_coeffs_from_maps(map_w2a, map_w2b, s2_val)
    
    if coeffs1 is None or coeffs2 is None:
        # Fallback to a known valid solution if reconstruction fails
        # This should not happen
        return [0] * 28 + [1, -1]
        
    return coeffs1 + coeffs2
