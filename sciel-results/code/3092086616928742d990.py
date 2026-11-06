import numpy as np
import time
from math import lcm

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5 second safety margin before 90s kill
    
    n = 30
    # Calculate LCM of 1..30
    L = 1
    for i in range(1, n + 1):
        L = lcm(L, i)
        
    # Weights
    weights = np.array([L // i for i in range(1, n + 1)], dtype=np.int64)
    
    half = n // 2
    w1 = weights[:half]
    w2 = weights[half:]
    
    # Generate sums and indices using vectorized approach
    def gen_sums_vec(w):
        k = len(w)
        # Start with just [0]
        sums = np.array([0], dtype=np.int64)
        idxs = np.array([0], dtype=np.int64)
        for i, weight in enumerate(w):
            pow3 = 3 ** i
            # For each existing sum, create 3 new ones: s-w, s, s+w
            # and corresponding indices: idx, idx+pow3, idx+2*pow3
            new_sums = np.concatenate([
                sums - weight,
                sums,
                sums + weight
            ])
            new_idxs = np.concatenate([
                idxs,
                idxs + pow3,
                idxs + 2 * pow3
            ])
            sums = new_sums
            idxs = new_idxs
        return sums, idxs

    # Generate first half
    sums1, idxs1 = gen_sums_vec(w1)
    
    # Sort first half by sum value
    sorted_order = np.argsort(sums1)
    sorted_sums1 = sums1[sorted_order]
    sorted_idxs1 = idxs1[sorted_order]
    
    # Generate second half
    sums2, idxs2 = gen_sums_vec(w2)
    
    best_abs = np.iinfo(np.int64).max
    best_idx1 = -1
    best_idx2 = -1
    
    # Vectorized search: for each s2, find closest s1 such that s1+s2 is closest to 0
    targets = -sums2
    # Find insertion positions
    positions = np.searchsorted(sorted_sums1, targets)
    
    # Check position and position-1 for each
    # Clip positions to valid range
    pos_clipped = np.clip(positions, 0, len(sorted_sums1) - 1)
    
    # Candidate 1: position (clipped)
    s1_cand1 = sorted_sums1[pos_clipped]
    idx1_cand1 = sorted_idxs1[pos_clipped]
    total1 = s1_cand1 + sums2
    abs1 = np.abs(total1)
    # Mask out zero sums
    mask1 = total1 != 0
    abs1 = np.where(mask1, abs1, np.iinfo(np.int64).max)
    
    # Candidate 2: position - 1 (clipped)
    pos2 = np.clip(positions - 1, 0, len(sorted_sums1) - 1)
    s1_cand2 = sorted_sums1[pos2]
    idx1_cand2 = sorted_idxs1[pos2]
    total2 = s1_cand2 + sums2
    abs2 = np.abs(total2)
    mask2 = total2 != 0
    abs2 = np.where(mask2, abs2, np.iinfo(np.int64).max)
    
    # Combine both candidates
    all_abs = np.minimum(abs1, abs2)
    all_idx1 = np.where(abs1 <= abs2, idx1_cand1, idx1_cand2)
    all_idx2 = np.arange(len(sums2))
    
    # Find the minimum
    min_pos = np.argmin(all_abs)
    best_abs_val = all_abs[min_pos]
    
    if best_abs_val < best_abs:
        best_abs = best_abs_val
        best_idx1 = all_idx1[min_pos]
        best_idx2 = all_idx2[min_pos]
    
    if best_idx1 == -1 or best_abs == np.iinfo(np.int64).max:
        # Fallback
        return [0]*28 + [1, -1]
    
    # Decode indices to coefficients
    def decode(idx, k):
        coeffs = []
        rem = int(idx)
        for j in range(k):
            digit = rem % 3
            rem //= 3
            if digit == 0:
                coeffs.append(-1)
            elif digit == 1:
                coeffs.append(0)
            else:
                coeffs.append(1)
        return coeffs
    
    c1 = decode(best_idx1, half)
    c2 = decode(best_idx2, n - half)
    
    return c1 + c2
