import time
import numpy as np
from math import gcd
from functools import reduce

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
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
    sums2_sorted = np.sort(sums2)
    
    len_s2 = len(sums2_sorted)
    
    # Precompute maps for reconstruction
    # Split halves into sub-halves for faster map building
    mid1 = len(w1) // 2
    mid2 = len(w2) // 2
    
    w1a = w1[:mid1]
    w1b = w1[mid1:]
    w2a = w2[:mid2]
    w2b = w2[mid2:]
    
    def build_sum_map(weights_sub):
        n_sub = len(weights_sub)
        sums_map = {0: []}
        for i, w in enumerate(weights_sub):
            w_int = int(w)
            new_sums_map = {}
            for s, coeffs in sums_map.items():
                ns = s - w_int
                new_coeffs = coeffs + [-1]
                if ns not in new_sums_map:
                    new_sums_map[ns] = new_coeffs
                ns = s
                new_coeffs = coeffs + [0]
                if ns not in new_sums_map:
                    new_sums_map[ns] = new_coeffs
                ns = s + w_int
                new_coeffs = coeffs + [1]
                if ns not in new_sums_map:
                    new_sums_map[ns] = new_coeffs
            sums_map = new_sums_map
        return sums_map

    map_w1a = build_sum_map(w1a)
    map_w1b = build_sum_map(w1b)
    map_w2a = build_sum_map(w2a)
    map_w2b = build_sum_map(w2b)
    
    def find_coeffs_from_maps(map_a, map_b, target_sum):
        target_sum = int(target_sum)
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

    # Targeted search for small numerators
    max_k = 10000
    found = False
    best_coeffs = None
    
    for k in range(1, max_k + 1):
        if time.time() > deadline - 5.0:
            break
            
        # Check for sum = k
        targets = k - sums1
        indices = np.searchsorted(sums2_sorted, targets)
        # Handle bounds
        idx_clipped = np.clip(indices, 0, len_s2 - 1)
        matched = sums2_sorted[idx_clipped] == targets
        # Also check index - 1 in case of duplicates or boundary
        idx_l = np.clip(indices - 1, 0, len_s2 - 1)
        matched_l = sums2_sorted[idx_l] == targets
        
        if np.any(matched) or np.any(matched_l):
            # Find the first matching index
            if np.any(matched):
                idx = np.where(matched)[0][0]
                s1_val = sums1[idx]
                s2_val = targets[idx]
            else:
                idx = np.where(matched_l)[0][0]
                s1_val = sums1[idx]
                s2_val = targets[idx]
            
            coeffs1 = find_coeffs_from_maps(map_w1a, map_w1b, s1_val)
            coeffs2 = find_coeffs_from_maps(map_w2a, map_w2b, s2_val)
            
            if coeffs1 is not None and coeffs2 is not None:
                best_coeffs = coeffs1 + coeffs2
                found = True
                break
        
        # Check for sum = -k
        targets = -k - sums1
        indices = np.searchsorted(sums2_sorted, targets)
        idx_clipped = np.clip(indices, 0, len_s2 - 1)
        matched = sums2_sorted[idx_clipped] == targets
        idx_l = np.clip(indices - 1, 0, len_s2 - 1)
        matched_l = sums2_sorted[idx_l] == targets
        
        if np.any(matched) or np.any(matched_l):
            if np.any(matched):
                idx = np.where(matched)[0][0]
                s1_val = sums1[idx]
                s2_val = targets[idx]
            else:
                idx = np.where(matched_l)[0][0]
                s1_val = sums1[idx]
                s2_val = targets[idx]
            
            coeffs1 = find_coeffs_from_maps(map_w1a, map_w1b, s1_val)
            coeffs2 = find_coeffs_from_maps(map_w2a, map_w2b, s2_val)
            
            if coeffs1 is not None and coeffs2 is not None:
                best_coeffs = coeffs1 + coeffs2
                found = True
                break

    if found:
        return best_coeffs

    # Fallback: Find global minimum non-zero sum
    # This part is similar to the original code but optimized
    targets = -sums1
    indices = np.searchsorted(sums2_sorted, targets)
    
    # Check neighbors
    candidates = []
    
    idx_r = np.clip(indices, 0, len_s2 - 1)
    s2_r = sums2_sorted[idx_r]
    total_r = sums1 + s2_r
    abs_r = np.abs(total_r)
    abs_r[total_r == 0] = np.iinfo(np.int64).max
    min_r_idx = np.argmin(abs_r)
    candidates.append((abs_r[min_r_idx], sums1[min_r_idx], s2_r[min_r_idx]))
    
    idx_l = np.clip(indices - 1, 0, len_s2 - 1)
    s2_l = sums2_sorted[idx_l]
    total_l = sums1 + s2_l
    abs_l = np.abs(total_l)
    abs_l[total_l == 0] = np.iinfo(np.int64).max
    min_l_idx = np.argmin(abs_l)
    candidates.append((abs_l[min_l_idx], sums1[min_l_idx], s2_l[min_l_idx]))
    
    idx_rr = np.clip(indices + 1, 0, len_s2 - 1)
    s2_rr = sums2_sorted[idx_rr]
    total_rr = sums1 + s2_rr
    abs_rr = np.abs(total_rr)
    abs_rr[total_rr == 0] = np.iinfo(np.int64).max
    min_rr_idx = np.argmin(abs_rr)
    candidates.append((abs_rr[min_rr_idx], sums1[min_rr_idx], s2_rr[min_rr_idx]))
    
    min_val, s1_val, s2_val = min(candidates)
    
    coeffs1 = find_coeffs_from_maps(map_w1a, map_w1b, s1_val)
    coeffs2 = find_coeffs_from_maps(map_w2a, map_w2b, s2_val)
    
    if coeffs1 is None or coeffs2 is None:
        return [0] * 28 + [1, -1]
        
    return coeffs1 + coeffs2
