import numpy as np
import math
import time

def construct():
    start_time = time.time()
    deadline = 50.0  # Safety margin
    
    n = 30
    # Calculate LCM of 1..30
    L = 1
    for i in range(1, n + 1):
        L = L * i // math.gcd(L, i)
    
    # a_i = L / i
    a = np.array([L // i for i in range(1, n + 1)], dtype=np.int64)
    
    half = n // 2
    a1 = a[:half]
    a2 = a[half:]
    
    def generate_sums(values):
        sums = np.array([0], dtype=np.int64)
        for v in values:
            # Order: -v, 0, +v
            sums = np.concatenate([sums - v, sums, sums + v])
        return sums

    # Generate sums for first half
    sums1 = generate_sums(a1)
    
    # Generate sums for second half
    sums2 = generate_sums(a2)
    
    # Sort sums1 to allow binary search
    # We need to keep track of original indices to decode coefficients
    perm1 = np.argsort(sums1)
    sorted_sums1 = sums1[perm1]
    
    # We want to find s1 in sorted_sums1 and s2 in sums2 such that |s1 + s2| is minimized and non-zero.
    # Iterate through sums2? No, vectorize.
    # For each s2, target is -s2.
    # Find insertion point in sorted_sums1.
    
    # Note: sums2 size is 3^15 ~ 14M.
    # searchsorted is fast.
    
    # We check the element at the insertion point and the one before it.
    
    # To avoid huge intermediate arrays, we can process in chunks or just do it all.
    # 14M elements is fine.
    
    targets = -sums2
    idxs = np.searchsorted(sorted_sums1, targets)
    
    # Candidates are at idxs and idxs - 1
    # Handle boundaries
    idxs_clipped = np.clip(idxs, 0, len(sorted_sums1) - 1)
    idxs_minus = np.clip(idxs - 1, 0, len(sorted_sums1) - 1)
    
    # Get corresponding sorted sums
    cand1 = sorted_sums1[idxs_clipped]
    cand2 = sorted_sums1[idxs_minus]
    
    # Calculate total sums
    total1 = cand1 + sums2
    total2 = cand2 + sums2
    
    # We want min |total| != 0
    abs1 = np.abs(total1)
    abs2 = np.abs(total2)
    
    # Mask zeros
    mask1 = abs1 > 0
    mask2 = abs2 > 0
    
    # We need to find the global minimum.
    # We can set zeros to inf
    abs1[~mask1] = np.iinfo(np.int64).max
    abs2[~mask2] = np.iinfo(np.int64).max
    
    min_abs = min(np.min(abs1), np.min(abs2))
    
    # Find which one gave the min
    if np.min(abs1) <= np.min(abs2):
        # Find index of min in abs1
        idx_in_abs1 = np.argmin(abs1)
        # Get the corresponding index in sorted_sums1
        idx_sorted = idxs_clipped[idx_in_abs1]
        # Map back to original index in sums1
        idx_orig1 = perm1[idx_sorted]
        idx_orig2 = idx_in_abs1
    else:
        idx_in_abs2 = np.argmin(abs2)
        idx_sorted = idxs_minus[idx_in_abs2]
        idx_orig1 = perm1[idx_sorted]
        idx_orig2 = idx_in_abs2
        
    # Decode idx_orig1 to coefficients for first half
    def decode_idx(idx, k):
        coeffs = [0] * k
        for j in range(k - 1, -1, -1):
            # The last processed element corresponds to the most significant digit in our concatenation scheme?
            # Let's re-verify.
            # S_1 from S_0: [ -a1, 0, a1 ]
            # Index 0 -> -a1 (coeff -1)
            # Index 1 -> 0 (coeff 0)
            # Index 2 -> a1 (coeff 1)
            #
            # S_2 from S_1:
            # [ S_1 - a2, S_1, S_1 + a2 ]
            # Block 0 (indices 0..2): coeff of a2 is -1. Index in S_1 is i % 3.
            # Block 1 (indices 3..5): coeff of a2 is 0.
            # Block 2 (indices 6..8): coeff of a2 is 1.
            #
            # So for index i in S_k:
            # d_k = i // (3^(k-1))
            # coeff_k = d_k - 1
            # i_prev = i % (3^(k-1))
            #
            # This means we decode from the last added element (a_k) down to the first (a_1).
            # The loop j from k-1 down to 0 corresponds to a_k down to a_1.
            # So coeffs[j] will be the coefficient for a_{j+1}.
            
            power = 3 ** j
            d = idx // power
            idx %= power
            coeffs[j] = d - 1
        return coeffs

    coeffs1 = decode_idx(idx_orig1, half)
    coeffs2 = decode_idx(idx_orig2, n - half)
    
    result = coeffs1 + coeffs2
    
    # Verify
    s = sum(c * a[i] for i, c in enumerate(result))
    if s == 0:
        # Fallback, should not happen if logic is correct
        return [0] * 28 + [1, -1]
        
    return result

if __name__ == "__main__":
    # Quick test
    res = construct()
    L = 1
    for i in range(1, 31):
        L = L * i // math.gcd(L, i)
    s = sum(res[i-1] / i for i in range(1, 31))
    print(f"Result: {res}")
    print(f"Sum: {s}")
    print(f"Num: {int(round(s * L))}")
