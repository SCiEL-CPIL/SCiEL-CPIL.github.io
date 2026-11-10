import time
import numpy as np

def construct():
    start_time = time.time()
    TIME_LIMIT = 90.0
    SAFETY_MARGIN = 5.0

    # Precompute modular inverses for denominators 1..10
    inv = np.zeros(11, dtype=np.int64)
    for i in range(1, 11):
        inv[i] = pow(i, -1, 101)

    # Precompute sum and size for every mask from 0 to 1023 using DP
    # mask_sum[mask] = sum of inverses mod 101
    # mask_size[mask] = number of set bits
    mask_sum = np.zeros(1024, dtype=np.int64)
    mask_size = np.zeros(1024, dtype=np.int64)

    # Vectorized DP: for each mask, find the least significant set bit
    # and use the precomputed value of mask without that bit
    for mask in range(1, 1024):
        lsb = mask & -mask
        i = lsb.bit_length() - 1  # index 0..9 corresponds to denominator i+1
        prev_mask = mask ^ lsb
        denom = i + 1
        mask_sum[mask] = (mask_sum[prev_mask] + inv[denom]) % 101
        mask_size[mask] = mask_size[prev_mask] + 1

    # For each residue, find the minimum size subset that maps to it
    min_size_for_residue = np.full(101, 10**9, dtype=np.int64)
    best_mask_for_residue = np.zeros(101, dtype=np.int64)

    for mask in range(1, 1024):
        s = mask_sum[mask]
        sz = mask_size[mask]
        if sz < min_size_for_residue[s]:
            min_size_for_residue[s] = sz
            best_mask_for_residue[s] = mask

    # Safety check: all residues should be covered
    for r in range(101):
        if min_size_for_residue[r] >= 10**9:
            min_size_for_residue[r] = 10
            best_mask_for_residue[r] = 1023  # fallback: all 10 denominators

    # Deadline guard
    if time.time() - start_time > TIME_LIMIT - SAFETY_MARGIN:
        pass  # Return what we have; this branch should never trigger

    # Construct the result lists
    result = [[] for _ in range(101)]
    for r in range(101):
        mask = int(best_mask_for_residue[r])
        terms = []
        for i in range(10):
            if mask & (1 << i):
                terms.append(i + 1)
        result[r] = terms

    return result
