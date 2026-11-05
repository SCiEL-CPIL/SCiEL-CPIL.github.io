"""
Find the minimal start s for five consecutive quadratic residues mod 1009.
Combines the boolean-array precomputation of the current program with the
vectorized NumPy scanning approach from inspiration 3, plus a deadline guard.
"""
import time
import numpy as np


def construct():
    t0 = time.time()
    deadline = t0 + 85.0  # safety margin under the 90s kill

    p = 1009
    run_len = 5

    # Build quadratic residue indicator and root map using NumPy for speed
    x = np.arange(1, p)
    residues = (x * x) % p

    # is_qr[r] is True if r is a non-zero quadratic residue mod p
    is_qr = np.zeros(p, dtype=bool)
    is_qr[residues] = True

    # Build root_map: for each residue r, store the smallest x with x*x % p == r
    # We iterate x from 1 upward, so the first x we see for each r is the smallest.
    root_map = {}
    for i in range(len(x)):
        r = int(residues[i])
        if r > 0 and r not in root_map:
            root_map[r] = int(x[i])

    # Vectorized scan: for each candidate s in 1..p-run_len, check if
    # is_qr[s], is_qr[s+1], ..., is_qr[s+4] are all True.
    # Use sliding window sum: a run of 5 consecutive True values starting at s
    # means the sum of is_qr[s:s+5] == 5.
    candidates = np.arange(1, p - run_len + 1)  # 1 to 1004

    # Build a prefix-sum of is_qr to compute sliding window sums efficiently
    # prefix[k] = sum(is_qr[0:k])
    prefix = np.zeros(p + 1, dtype=np.int64)
    np.cumsum(is_qr.astype(np.int64), out=prefix[1:])

    # For candidate s, sum of is_qr[s:s+5] = prefix[s+5] - prefix[s]
    window_sums = prefix[candidates + run_len] - prefix[candidates]
    valid_mask = (window_sums == run_len)

    if np.any(valid_mask):
        # Find the first (smallest) valid s
        first_idx = np.argmax(valid_mask)  # argmax returns first True index
        s = int(candidates[first_idx])

        if time.time() > deadline:
            # Should not happen at this point, but guard just in case
            pass

        roots = []
        for i in range(run_len):
            val = s + i
            roots.append(root_map[val])

        return {'start': s, 'roots': roots}

    # Fallback: linear scan (defensive, should not be reached)
    for s in range(1, p - run_len + 1):
        if time.time() > deadline:
            break
        if all(is_qr[s + i] for i in range(run_len)):
            roots = [root_map[s + i] for i in range(run_len)]
            return {'start': s, 'roots': roots}

    raise RuntimeError('No run of 5 consecutive non-zero quadratic residues found')
