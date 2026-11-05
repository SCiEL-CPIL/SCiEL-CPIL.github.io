"""
Find the minimal start s for five consecutive non-zero quadratic residues mod 1009.
Returns {'start': s, 'roots': [r0, r1, r2, r3, r4]} where ri^2 mod 1009 == s+i.
Uses vectorized NumPy precomputation for speed.
"""
import numpy as np


def construct():
    p = 1009
    run_len = 5

    # x ranges from 1 to p-1 = 1008
    x = np.arange(1, p)
    residues = (x * x) % p

    # is_qr[r] is True if r is a non-zero quadratic residue mod p
    is_qr = np.zeros(p, dtype=bool)
    is_qr[residues] = True

    # Build root_arr: for each residue r in 1..1008, store the smallest x with x*x % p == r
    # x[i] = i+1, residues[i] = x[i]^2 % p
    # For each unique residue value, the first occurrence has the smallest x
    unique_res, first_idx = np.unique(residues, return_index=True)
    # first_idx[k] is index i in x where unique_res[k] first appears; root = x[i] = i+1
    root_arr = np.zeros(p, dtype=np.int32)
    root_arr[unique_res] = first_idx + 1  # x values are 1..1008

    # Sliding window sum via prefix sums to find runs of 5 consecutive QRs
    prefix = np.zeros(p + 1, dtype=np.int64)
    np.cumsum(is_qr.astype(np.int64), out=prefix[1:])

    candidates = np.arange(1, p - run_len + 1)  # 1 to 1004
    window_sums = prefix[candidates + run_len] - prefix[candidates]
    valid_mask = (window_sums == run_len)

    if not np.any(valid_mask):
        raise RuntimeError('No run of 5 consecutive non-zero quadratic residues found')

    first_idx_valid = int(np.argmax(valid_mask))
    s = int(candidates[first_idx_valid])

    roots = [int(root_arr[s + i]) for i in range(run_len)]

    return {'start': s, 'roots': roots}
