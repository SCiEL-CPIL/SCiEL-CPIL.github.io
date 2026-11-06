import numpy as np
from math import gcd

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    # Calculate LCM of 1 to 30
    L = 1
    for i in range(1, 31):
        L = _lcm(L, i)
        
    # Weights w_i = L / (i+1) for i=0..29
    weights = [L // (i + 1) for i in range(30)]
    
    # Split into two halves
    w1 = weights[:15]
    w2 = weights[15:]
    
    def generate_sums(weights_half):
        # Initialize with sum 0 and code 0
        sums = np.array([0], dtype=np.int64)
        codes = np.array([0], dtype=np.int32)
        
        for k, w in enumerate(weights_half):
            pow3 = 3 ** k
            # Create three copies: -w, 0, +w
            # Corresponding code changes: -3^k, 0, +3^k
            
            # To save memory, we can do this in place if we pre-allocate,
            # but concatenate is easier and fast enough for N=15.
            
            sums_new = np.concatenate([sums - w, sums, sums + w])
            codes_new = np.concatenate([codes - pow3, codes, codes + pow3])
            
            sums = sums_new
            codes = codes_new
            
        return sums, codes

    # Generate sums for first half
    sums1, codes1 = generate_sums(w1)
    # Generate sums for second half
    sums2, codes2 = generate_sums(w2)
    
    # Sort sums1 and keep codes1 aligned
    idx1 = np.argsort(sums1)
    sums1_sorted = sums1[idx1]
    codes1_sorted = codes1[idx1]
    
    # Sort sums2 and keep codes2 aligned
    idx2 = np.argsort(sums2)
    sums2_sorted = sums2[idx2]
    codes2_sorted = codes2[idx2]
    
    # Vectorized search for closest sum to 0
    # We want to minimize |s1 + s2| subject to s1 + s2 != 0
    
    targets = -sums1_sorted
    # Find index where target would be inserted
    idxs = np.searchsorted(sums2_sorted, targets)
    
    # We need to check idxs and idxs-1
    # Clip idxs to valid range
    idxs_clipped = np.clip(idxs, 0, len(sums2_sorted) - 1)
    idxs_minus = np.clip(idxs - 1, 0, len(sums2_sorted) - 1)
    
    # Calculate sums for candidates
    s1 = sums1_sorted
    s2_a = sums2_sorted[idxs_clipped]
    s2_b = sums2_sorted[idxs_minus]
    
    total_sums_a = s1 + s2_a
    total_sums_b = s1 + s2_b
    
    # Combine candidates
    all_sums = np.concatenate([total_sums_a, total_sums_b])
    all_codes1 = np.concatenate([codes1_sorted, codes1_sorted])
    all_codes2 = np.concatenate([codes2_sorted[idxs_clipped], codes2_sorted[idxs_minus]])
    
    # Filter out zero sums
    mask = all_sums != 0
    all_sums = all_sums[mask]
    all_codes1 = all_codes1[mask]
    all_codes2 = all_codes2[mask]
    
    if len(all_sums) == 0:
        # Fallback, though unlikely
        return [0] * 28 + [1, -1]
        
    # Find minimum absolute value
    abs_sums = np.abs(all_sums)
    min_idx = np.argmin(abs_sums)
    
    best_code1 = all_codes1[min_idx]
    best_code2 = all_codes2[min_idx]
    
    # Decode codes to coefficients
    coeffs = [0] * 30
    
    # Decode code1 (first 15 coefficients)
    temp = best_code1
    for k in range(15):
        digit = temp % 3
        if digit == 0:
            coeffs[k] = 0
        elif digit == 1:
            coeffs[k] = 1
        elif digit == 2:
            coeffs[k] = -1
        temp = temp // 3
        
    # Decode code2 (last 15 coefficients)
    temp = best_code2
    for k in range(15):
        digit = temp % 3
        if digit == 0:
            coeffs[15 + k] = 0
        elif digit == 1:
            coeffs[15 + k] = 1
        elif digit == 2:
            coeffs[15 + k] = -1
        temp = temp // 3
        
    return coeffs
