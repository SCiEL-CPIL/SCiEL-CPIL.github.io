import numpy as np
import time
from collections import defaultdict

LIMIT = 10000000

def construct():
    deadline = time.time() + 80.0

    # Step 1: Compute sigma(n) for n = 1..LIMIT using NumPy divisor-sum sieve
    sigma = np.zeros(LIMIT + 1, dtype=np.int64)
    for d in range(1, LIMIT + 1):
        sigma[d::d] += d

    if time.time() > deadline:
        return [11, 6]

    # Step 2: Group numbers by their sigma value
    groups = defaultdict(list)
    for n in range(2, LIMIT + 1):
        s = int(sigma[n])
        groups[s].append(n)

    if time.time() > deadline:
        return [11, 6]

    # Step 3: For each sigma-fiber with >= 2 members, find the pair (a,b)
    # minimizing |a/b - 3/2| = |2a - 3b| / (2b), with a != b and 2a != 3b.
    best_a, best_b = 11, 6
    best_err = float('inf')

    for s, lst in groups.items():
        if time.time() > deadline:
            break
        if len(lst) < 2:
            continue
        arr = np.array(lst, dtype=np.int64)

        # Direction 1: treat each element as b, find a closest to 1.5*b
        ideal_a = 1.5 * arr.astype(np.float64)
        idx = np.searchsorted(arr, ideal_a)

        for offset in (0, -1):
            jj = np.clip(idx + offset, 0, len(arr) - 1)
            a_vals = arr[jj]
            b_vals = arr
            mask = a_vals != b_vals
            if not np.any(mask):
                continue
            a_vals = a_vals[mask]
            b_vals = b_vals[mask]
            num = np.abs(2 * a_vals - 3 * b_vals)
            mask2 = num > 0
            if not np.any(mask2):
                continue
            a_vals = a_vals[mask2]
            b_vals = b_vals[mask2]
            num = num[mask2]
            errs = num / (2.0 * b_vals.astype(np.float64))
            min_idx = int(np.argmin(errs))
            e = float(errs[min_idx])
            if e < best_err:
                best_err = e
                best_a = int(a_vals[min_idx])
                best_b = int(b_vals[min_idx])

        # Direction 2: treat each element as a, find b closest to (2/3)*a
        ideal_b = (2.0 / 3.0) * arr.astype(np.float64)
        idx2 = np.searchsorted(arr, ideal_b)

        for offset in (0, -1):
            jj = np.clip(idx2 + offset, 0, len(arr) - 1)
            b_vals = arr[jj]
            a_vals = arr
            mask = a_vals != b_vals
            if not np.any(mask):
                continue
            a_vals = a_vals[mask]
            b_vals = b_vals[mask]
            num = np.abs(2 * a_vals - 3 * b_vals)
            mask2 = num > 0
            if not np.any(mask2):
                continue
            a_vals = a_vals[mask2]
            b_vals = b_vals[mask2]
            num = num[mask2]
            errs = num / (2.0 * b_vals.astype(np.float64))
            min_idx = int(np.argmin(errs))
            e = float(errs[min_idx])
            if e < best_err:
                best_err = e
                best_a = int(a_vals[min_idx])
                best_b = int(b_vals[min_idx])

    if best_err == float('inf'):
        return [11, 6]

    return [int(best_a), int(best_b)]
