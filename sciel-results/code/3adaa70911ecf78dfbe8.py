import time
import numpy as np
from math import gcd
from functools import reduce

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 90s limit

    n = 30

    def lcm(a, b):
        return a * b // gcd(a, b)

    L = reduce(lcm, range(1, n + 1))

    # W[i] = L // (i+1), so sum(c[i]/(i+1)) = (1/L) * sum(c[i]*W[i])
    # We want to minimize |sum(c[i]*W[i])| subject to it being nonzero
    W = [L // (i + 1) for i in range(n)]

    # Split into two halves for meet-in-the-middle
    mid = n // 2
    w1 = W[:mid]
    w2 = W[mid:]

    def generate_sums(weights_half):
        """
        Generate all possible sums for a half using coefficients in {-1, 0, 1}.
        Returns a numpy array of sums.
        """
        sums = np.array([0], dtype=np.int64)
        for w in weights_half:
            # Vectorized expansion: for each existing sum s, we get s-w, s, s+w
            sums = np.concatenate([sums - w, sums, sums + w])
        return sums

    def find_coeffs(weights, target):
        """
        Find coefficients in {-1, 0, 1} for the given weights such that
        sum(c[i] * weights[i]) == target.
        Returns the list of coefficients or None if not found.
        """
        n_half = len(weights)
        coeffs = [0] * n_half
        
        def backtrack(idx, current_sum):
            if idx == n_half:
                return current_sum == target
            w = weights[idx]
            # Try -1, 0, 1
            for c in (-1, 0, 1):
                coeffs[idx] = c
                if backtrack(idx + 1, current_sum + c * w):
                    return True
            return False

        if backtrack(0, 0):
            return coeffs
        return None

    # Generate sums for both halves
    # 3^15 = 14,348,907 elements.
    # Memory: ~115 MB per array. Total ~230 MB. Acceptable.
    sums1 = generate_sums(w1)
    sums2 = generate_sums(w2)
    
    # Sort sums2 for binary search
    sums2_sorted = np.sort(sums2)
    
    best_abs_val = float('inf')
    best_s1 = 0
    best_s2 = 0
    
    # To speed up, we can iterate over unique sums in sums1 if there are many duplicates,
    # but with random-ish large weights, duplicates are rare.
    # However, 14 million iterations in a Python loop is slow.
    # We can vectorize the search part.
    
    # We want to find pairs (s1, s2) such that |s1 + s2| is minimized and non-zero.
    # For each s1, we want s2 close to -s1.
    
    # Vectorized search:
    targets = -sums1
    indices = np.searchsorted(sums2_sorted, targets)
    
    len_s2 = len(sums2_sorted)
    
    # We need to check indices and indices-1 (and maybe indices+1 if index is at boundary or value matches exactly?)
    # searchsorted returns the index where the value would be inserted to maintain order.
    # If the value exists, it returns the leftmost index.
    # So we should check:
    # 1. The element at `indices` (if valid)
    # 2. The element at `indices - 1` (if valid)
    
    # Let's create candidate arrays
    # Candidate 1: sums2_sorted[indices]
    idx_r = np.clip(indices, 0, len_s2 - 1)
    s2_r = sums2_sorted[idx_r]
    total_r = sums1 + s2_r
    
    # Candidate 2: sums2_sorted[indices - 1]
    idx_l = np.clip(indices - 1, 0, len_s2 - 1)
    s2_l = sums2_sorted[idx_l]
    total_l = sums1 + s2_l
    
    # We want the min |total| for total != 0.
    # Mask out zeros
    mask_r = total_r != 0
    mask_l = total_l != 0
    
    abs_total_r = np.abs(total_r)
    abs_total_l = np.abs(total_l)
    
    # Set zero candidates to infinity
    abs_total_r[~mask_r] = np.iinfo(np.int64).max
    abs_total_l[~mask_l] = np.iinfo(np.int64).max
    
    # Find the global minimum
    min_abs_r = np.min(abs_total_r)
    min_abs_l = np.min(abs_total_l)
    
    if min_abs_r < min_abs_l:
        best_abs_val = min_abs_r
        idx_best = np.argmin(abs_total_r)
        best_s1 = sums1[idx_best]
        best_s2 = s2_r[idx_best]
    else:
        best_abs_val = min_abs_l
        idx_best = np.argmin(abs_total_l)
        best_s1 = sums1[idx_best]
        best_s2 = s2_l[idx_best]
        
    # Check if we found a valid non-zero sum
    if best_abs_val == np.iinfo(np.int64).max:
        # Fallback: This should not happen, but just in case
        # Try to find any non-zero sum
        # If all sums are 0, it means L=0 or something weird, but L is LCM so it's fine.
        # It might happen if all combinations result in 0? Impossible for n=30.
        # Let's just pick the first non-zero from the list
        for i in range(len(sums1)):
            if sums1[i] != 0:
                best_s1 = sums1[i]
                best_s2 = 0 # Pair with 0 from sums2? No, 0 is in sums2.
                # We need a valid pair from our search.
                # If min was inf, it means all candidates were 0.
                # This implies sums1 + sums2 is always 0? No.
                # It implies for every s1, the closest s2 made s1+s2=0.
                # This is highly unlikely.
                pass
        # If we still don't have a best, return a default non-zero
        # e.g., c = [1, 0, ..., 0] -> sum = 1/1 = 1. N = L.
        return [1] + [0]*29

    # Reconstruct coefficients
    coeffs1 = find_coeffs(w1, int(best_s1))
    coeffs2 = find_coeffs(w2, int(best_s2))
    
    if coeffs1 is None or coeffs2 is None:
        # This should not happen if best_s1 and best_s2 were generated from these weights
        # Fallback
        return [1] + [0]*29
        
    coeffs = coeffs1 + coeffs2
    
    # Verify
    # val = sum(coeffs[i]/(i+1) for i in range(n))
    # assert val != 0
    # assert abs(val) == Fraction(best_abs_val, L)
    
    return coeffs
