import math
import time
import numpy as np
from fractions import Fraction

def construct():
    """
    Returns exactly 30 integer coefficients from {-1, 0, 1} such that
    sum(coeff[i-1]/i for i in 1..30) is nonzero and has minimal absolute value.
    """
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit

    n = 30
    half = 15

    # Calculate LCM of 1 to 30
    def lcm(a, b):
        return a // math.gcd(a, b) * b

    L = 1
    for i in range(1, n + 1):
        L = lcm(L, i)

    # Scaled weights: weight[i] = L // (i+1)
    weights = [L // (i + 1) for i in range(n)]
    w1 = weights[:half]
    w2 = weights[half:]

    def gen_sums_with_indices(w):
        """
        Generate all 3^len(w) possible sums along with their index patterns.
        Returns (sums_array, index_patterns_array) where index_patterns encodes
        the coefficient choices as integers (0 for -1, 1 for 0, 2 for +1).
        """
        sums = np.array([0], dtype=np.int64)
        patterns = np.array([0], dtype=np.int64)
        base = 1
        for wi in w:
            new_sums = np.empty(3 * len(sums), dtype=np.int64)
            new_patterns = np.empty(3 * len(sums), dtype=np.int64)
            new_sums[0:len(sums)] = sums - wi
            new_sums[len(sums):2*len(sums)] = sums
            new_sums[2*len(sums):3*len(sums)] = sums + wi
            new_patterns[0:len(sums)] = patterns + 0 * base
            new_patterns[len(sums):2*len(sums)] = patterns + 1 * base
            new_patterns[2*len(sums):3*len(sums)] = patterns + 2 * base
            sums = new_sums
            patterns = new_patterns
            base *= 3
        return sums, patterns

    # Generate all sums and patterns for both halves
    sums1, patterns1 = gen_sums_with_indices(w1)
    sums2, patterns2 = gen_sums_with_indices(w2)

    if time.time() > deadline:
        return [0] * 28 + [1, -1]

    # Sort sums2 and track original indices
    sort_idx = np.argsort(sums2)
    sums2_sorted = sums2[sort_idx]
    patterns2_sorted = patterns2[sort_idx]

    # Vectorized search for minimum |s1 + s2| where s1+s2 != 0
    targets = -sums1
    indices = np.searchsorted(sums2_sorted, targets)

    len2 = len(sums2_sorted)
    max_int64 = np.iinfo(np.int64).max

    # Candidate 1: sums2_sorted[indices] (first element >= target)
    idx1 = indices
    mask1 = (idx1 >= 0) & (idx1 < len2)
    cand1_s2 = np.zeros(len(sums1), dtype=np.int64)
    cand1_s2[mask1] = sums2_sorted[idx1[mask1]]
    cand1_pat = np.zeros(len(sums1), dtype=np.int64)
    cand1_pat[mask1] = patterns2_sorted[idx1[mask1]]
    total1 = sums1 + cand1_s2
    valid1 = mask1 & (total1 != 0)
    abs1 = np.where(valid1, np.abs(total1), max_int64)

    # Candidate 2: sums2_sorted[indices-1] (last element < target)
    idx2 = indices - 1
    mask2 = (idx2 >= 0) & (idx2 < len2)
    cand2_s2 = np.zeros(len(sums1), dtype=np.int64)
    cand2_s2[mask2] = sums2_sorted[idx2[mask2]]
    cand2_pat = np.zeros(len(sums1), dtype=np.int64)
    cand2_pat[mask2] = patterns2_sorted[idx2[mask2]]
    total2 = sums1 + cand2_s2
    valid2 = mask2 & (total2 != 0)
    abs2 = np.where(valid2, np.abs(total2), max_int64)

    # Find global minimum
    min_abs1 = np.min(abs1)
    min_abs2 = np.min(abs2)

    if min_abs1 < min_abs2:
        best_val = min_abs1
        idx_min = np.argmin(abs1)
        best_s1 = sums1[idx_min]
        best_s2 = cand1_s2[idx_min]
        best_pat1 = patterns1[idx_min]
        best_pat2 = cand1_pat[idx_min]
    else:
        best_val = min_abs2
        idx_min = np.argmin(abs2)
        best_s1 = sums1[idx_min]
        best_s2 = cand2_s2[idx_min]
        best_pat1 = patterns1[idx_min]
        best_pat2 = cand2_pat[idx_min]

    if time.time() > deadline:
        return [0] * 28 + [1, -1]

    def decode_pattern(pattern, length):
        """Decode an integer pattern back to coefficients in {-1, 0, 1}."""
        coeffs = [0] * length
        for i in range(length):
            digit = pattern % 3
            pattern //= 3
            coeffs[i] = digit - 1  # 0->-1, 1->0, 2->1
        return coeffs

    c1 = decode_pattern(int(best_pat1), half)
    c2 = decode_pattern(int(best_pat2), half)

    coeffs = c1 + c2

    # Final verification
    total_frac = sum(Fraction(c, i + 1) for i, c in enumerate(coeffs))
    if total_frac == 0:
        return [0] * 28 + [1, -1]

    return coeffs

# EVOLVE-BLOCK-END
