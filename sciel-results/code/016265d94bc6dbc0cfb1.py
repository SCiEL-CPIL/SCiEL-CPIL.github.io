import time
from math import gcd
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = 30
    
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = 1
    for i in range(1, n + 1):
        L = lcm(L, i)
        
    weights = np.array([L // i for i in range(1, n + 1)], dtype=np.int64)
    
    def get_group_sums(start, end):
        size = end - start
        group_weights = weights[start:end]
        # Generate all combinations of {-1, 0, 1} for 'size' elements
        # Using meshgrid is efficient for small sizes
        coeffs_list = np.array(np.meshgrid(*[[-1, 0, 1]] * size)).T.reshape(-1, size)
        sums = coeffs_list @ group_weights
        return sums, coeffs_list
    
    # Split into 5 groups: 7, 7, 7, 7, 2
    G1_sums, G1_coeffs = get_group_sums(0, 7)
    G2_sums, G2_coeffs = get_group_sums(7, 14)
    G3_sums, G3_coeffs = get_group_sums(14, 21)
    G4_sums, G4_coeffs = get_group_sums(21, 28)
    G5_sums, G5_coeffs = get_group_sums(28, 30)
    
    # Combine G1 and G2
    # S12[i,j] = G1_sums[i] + G2_sums[j]
    # Flatten to 1D
    S12_sums = (G1_sums[:, None] + G2_sums[None, :]).flatten()
    S12_sort_idx = np.argsort(S12_sums)
    S12_sorted = S12_sums[S12_sort_idx]
    
    # Combine G3 and G4
    S34_sums = (G3_sums[:, None] + G4_sums[None, :]).flatten()
    S34_sort_idx = np.argsort(S34_sums)
    S34_sorted = S34_sums[S34_sort_idx]
    
    len_S34 = len(S34_sorted)
    
    best_abs = None
    best_val = None
    best_i12 = None # Index in S12_sorted
    best_j34 = None # Index in S34_sorted
    best_s5_idx = None
    
    for s5_idx in range(len(G5_sums)):
        target = -G5_sums[s5_idx]
        
        # We want min |S12_sorted[i] + S34_sorted[j] - target| != 0
        
        # Vectorized binary search
        # For each x in S12_sorted, find y in S34_sorted closest to target - x
        targets_for_s34 = target - S12_sorted
        idxs = np.searchsorted(S34_sorted, targets_for_s34)
        
        # Candidate 1: S34_sorted[idxs]
        # Clip to valid range [0, len-1]
        idxs_clipped = np.clip(idxs, 0, len_S34 - 1)
        cands1 = S12_sorted + S34_sorted[idxs_clipped] - target
        
        # Candidate 2: S34_sorted[idxs - 1]
        idxs_minus1 = idxs - 1
        valid2 = idxs_minus1 >= 0
        idxs_minus1_clipped = np.clip(idxs_minus1, 0, len_S34 - 1)
        cands2 = S12_sorted + S34_sorted[idxs_minus1_clipped] - target
        # Invalidate cands2 where idxs was 0
        cands2[~valid2] = np.inf
        
        # Exclude zero sums
        abs_cands1 = np.abs(cands1)
        abs_cands2 = np.abs(cands2)
        
        abs_cands1[cands1 == 0] = np.inf
        abs_cands2[cands2 == 0] = np.inf
        
        min1 = np.min(abs_cands1)
        idx1 = np.argmin(abs_cands1)
        
        min2 = np.min(abs_cands2)
        idx2 = np.argmin(abs_cands2)
        
        if min1 < min2:
            curr_abs = min1
            curr_val = cands1[idx1]
            curr_i12 = idx1
            curr_j34 = idxs_clipped[idx1]
        else:
            curr_abs = min2
            curr_val = cands2[idx2]
            curr_i12 = idx2
            curr_j34 = idxs_minus1_clipped[idx2]
            
        if best_abs is None or curr_abs < best_abs:
            best_abs = curr_abs
            best_val = curr_val
            best_i12 = curr_i12
            best_j34 = curr_j34
            best_s5_idx = s5_idx
            
        if time.time() > deadline:
            break

    if best_i12 is None:
        # Fallback
        fallback = [0] * 30
        fallback[1] = 1
        fallback[2] = -1
        return fallback

    # Map sorted indices back to original flat indices
    s12_flat_idx = S12_sort_idx[best_i12]
    s34_flat_idx = S34_sort_idx[best_j34]
    
    # Decompose flat indices into group indices
    len_G2 = len(G2_sums)
    len_G4 = len(G4_sums)
    
    g1_idx = s12_flat_idx // len_G2
    g2_idx = s12_flat_idx % len_G2
    
    g3_idx = s34_flat_idx // len_G4
    g4_idx = s34_flat_idx % len_G4
    
    coeffs = np.concatenate([
        G1_coeffs[g1_idx], 
        G2_coeffs[g2_idx], 
        G3_coeffs[g3_idx], 
        G4_coeffs[g4_idx], 
        G5_coeffs[best_s5_idx]
    ])
    
    return [int(c) for c in coeffs]
