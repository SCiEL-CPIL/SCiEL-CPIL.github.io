import time
import numpy as np
from math import gcd
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin
    
    n = 30
    L = 1
    for i in range(1, n + 1):
        L = L * i // gcd(L, i)
        
    weights = [L // (i + 1) for i in range(n)]
    
    half = n // 2
    left_weights = weights[:half]
    right_weights = weights[half:]
    
    def gen_sums_paths(weights):
        """
        Generates all possible sums and their path IDs.
        Path ID is an integer representing the base-3 digits of the coefficients.
        Digit 0 -> coeff 0
        Digit 1 -> coeff 1
        Digit 2 -> coeff -1
        """
        sums = [0]
        paths = [0]
        power = 1
        for w in weights:
            new_sums = []
            new_paths = []
            # Pre-allocate? No, list append is fine for 14M
            for s, p in zip(sums, paths):
                # coeff 0
                new_sums.append(s)
                new_paths.append(p)
                # coeff 1
                new_sums.append(s + w)
                new_paths.append(p + 1 * power)
                # coeff -1
                new_sums.append(s - w)
                new_paths.append(p + 2 * power)
            sums = new_sums
            paths = new_paths
            power *= 3
        return np.array(sums, dtype=np.int64), np.array(paths, dtype=np.int64)

    # Check time
    if time.time() > deadline:
        return [0] * 28 + [1, -1]

    left_sums, left_paths = gen_sums_paths(left_weights)
    right_sums, right_paths = gen_sums_paths(right_weights)
    
    if time.time() > deadline:
        return [0] * 28 + [1, -1]

    # Sort left sums and keep track of paths
    left_order = np.argsort(left_sums)
    sorted_left_sums = left_sums[left_order]
    sorted_left_paths = left_paths[left_order]
    
    # Vectorized search
    targets = -right_sums
    # searchsorted returns index in sorted_left_sums
    idx = np.searchsorted(sorted_left_sums, targets)
    
    # We need to check idx and idx-1
    # Clip indices to valid range [1, N-1] to safely access idx-1 and idx
    N = len(sorted_left_sums)
    idx_safe = np.clip(idx, 1, N - 1)
    
    # Candidate 1: idx_safe - 1
    cand1_sums = sorted_left_sums[idx_safe - 1]
    total1 = cand1_sums + right_sums
    abs1 = np.abs(total1)
    abs1[total1 == 0] = np.inf
    
    # Candidate 2: idx_safe
    cand2_sums = sorted_left_sums[idx_safe]
    total2 = cand2_sums + right_sums
    abs2 = np.abs(total2)
    abs2[total2 == 0] = np.inf
    
    min_abs1 = np.min(abs1)
    min_abs2 = np.min(abs2)
    
    if min_abs1 < min_abs2:
        best_right_idx = np.argmin(abs1)
        best_left_idx_in_sorted = idx_safe[best_right_idx] - 1
        best_path_left = sorted_left_paths[best_left_idx_in_sorted]
        best_path_right = right_paths[best_right_idx]
    else:
        best_right_idx = np.argmin(abs2)
        best_left_idx_in_sorted = idx_safe[best_right_idx]
        best_path_left = sorted_left_paths[best_left_idx_in_sorted]
        best_path_right = right_paths[best_right_idx]
        
    # Decode paths
    def decode_path(path_id, n_weights):
        coeffs = [0] * n_weights
        for i in range(n_weights):
            d = path_id % 3
            if d == 0:
                coeffs[i] = 0
            elif d == 1:
                coeffs[i] = 1
            else:
                coeffs[i] = -1
            path_id //= 3
        return coeffs

    left_coeffs = decode_path(best_path_left, half)
    right_coeffs = decode_path(best_path_right, n - half)
    
    result = left_coeffs + right_coeffs
    
    # Verify
    total_frac = sum(Fraction(c, i + 1) for i, c in enumerate(result) if c != 0)
    if total_frac == 0:
        # Fallback
        return [0] * 28 + [1, -1]
        
    return result

# EVOLVE-BLOCK-END
