import numpy as np
import time
from math import gcd
from functools import reduce

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit

    n = 30
    half = 15

    # 1. Calculate LCM of 1..30
    def lcm(a, b):
        return a * b // gcd(a, b)

    L = reduce(lcm, range(1, n + 1))

    # Precompute scaled terms
    terms = np.array([L // (i + 1) for i in range(n)], dtype=np.int64)

    size = 3 ** half

    def gen_sums_with_idx(terms_part):
        sums = np.array([0], dtype=np.int64)
        idx = np.array([0], dtype=np.int64)

        powers = 3 ** np.arange(len(terms_part))

        for i, t in enumerate(terms_part):
            curr_size = len(sums)
            new_sums = np.empty(curr_size * 3, dtype=np.int64)
            new_idx = np.empty(curr_size * 3, dtype=np.int64)

            # Digit 0 (coeff -1)
            new_sums[0:curr_size] = sums - t
            new_idx[0:curr_size] = idx

            # Digit 1 (coeff 0)
            new_sums[curr_size:2*curr_size] = sums
            new_idx[curr_size:2*curr_size] = idx + powers[i]

            # Digit 2 (coeff 1)
            new_sums[2*curr_size:3*curr_size] = sums + t
            new_idx[2*curr_size:3*curr_size] = idx + 2 * powers[i]

            sums = new_sums
            idx = new_idx

        return sums, idx

    # Generate for A (terms 0-14)
    sums_A, idx_A = gen_sums_with_idx(terms[0:15])

    if time.time() > deadline:
        return [1, -1] + [0]*28

    # Generate for B (terms 15-29)
    sums_B, idx_B = gen_sums_with_idx(terms[15:30])

    if time.time() > deadline:
        return [1, -1] + [0]*28

    # Sort sums_B and reorder idx_B
    sort_idx = np.argsort(sums_B)
    sums_B_sorted = sums_B[sort_idx]
    idx_B_sorted = idx_B[sort_idx]

    # Search for best pair
    # We want min |s_A + s_B| != 0

    targets = -sums_A
    pos = np.searchsorted(sums_B_sorted, targets)

    # Candidate 1: sums_B_sorted[pos]
    pos1 = np.clip(pos, 0, len(sums_B_sorted) - 1)
    cand_sums_B_1 = sums_B_sorted[pos1]
    total_1 = sums_A + cand_sums_B_1

    # Candidate 2: sums_B_sorted[pos-1]
    pos2 = np.clip(pos - 1, 0, len(sums_B_sorted) - 1)
    cand_sums_B_2 = sums_B_sorted[pos2]
    total_2 = sums_A + cand_sums_B_2

    # Two-pass min without concatenation
    # Pass 1
    abs_total_1 = np.abs(total_1)
    mask_1 = total_1 != 0
    abs_total_1[~mask_1] = np.iinfo(np.int64).max
    best_1 = int(np.argmin(abs_total_1))
    best_val_1 = abs_total_1[best_1]

    # Pass 2
    abs_total_2 = np.abs(total_2)
    mask_2 = total_2 != 0
    abs_total_2[~mask_2] = np.iinfo(np.int64).max
    best_2 = int(np.argmin(abs_total_2))
    best_val_2 = abs_total_2[best_2]

    if best_val_1 <= best_val_2:
        best_idx_A = best_1
        best_idx_B = int(pos1[best_1])
    else:
        best_idx_A = best_2
        best_idx_B = int(pos2[best_2])

    # Recover coefficients
    coeff_idx_A = int(idx_A[best_idx_A])
    coeff_idx_B = int(idx_B_sorted[best_idx_B])

    coeffs = np.zeros(30, dtype=int)

    powers = 3 ** np.arange(15)

    # Decode A
    digits_A = (coeff_idx_A // powers) % 3
    coeffs[0:15] = np.where(digits_A == 0, -1, np.where(digits_A == 1, 0, 1))

    # Decode B
    digits_B = (coeff_idx_B // powers) % 3
    coeffs[15:30] = np.where(digits_B == 0, -1, np.where(digits_B == 1, 0, 1))

    # Return as Python integers
    return [int(x) for x in coeffs]
