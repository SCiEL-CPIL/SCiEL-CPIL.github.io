import numpy as np
import time
import itertools

D = 11

def construct():
    deadline = time.time() + 140

    # Generate candidate vectors of various types
    candidates = []

    # Type 1: (±2, ±2, 0, ..., 0) — norm² = 8
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i] = 2 * si
                v[j] = 2 * sj
                candidates.append(v)

    # Type 2: (±1, ±1, 0, ..., 0) — norm² = 2
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i] = si
                v[j] = sj
                candidates.append(v)

    # Type 3: (±1, ±1, ±1, 0, ..., 0) — norm² = 3
    for i, j, k in itertools.combinations(range(D), 3):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    v = [0] * D
                    v[i] = si
                    v[j] = sj
                    v[k] = sk
                    candidates.append(v)

    # Type 4: (±1, ±1, ±1, ±1, 0, ..., 0) — norm² = 4
    for i, j, k, l in itertools.combinations(range(D), 4):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    for sl in (1, -1):
                        v = [0] * D
                        v[i] = si
                        v[j] = sj
                        v[k] = sk
                        v[l] = sl
                        candidates.append(v)

    # Type 5: (±1, ..., ±1) all 11 coords — norm² = 11
    for mask in range(2**D):
        v = [1 if (mask >> i) & 1 == 0 else -1 for i in range(D)]
        candidates.append(v)

    arr = np.array(candidates, dtype=np.int64)
    n = len(arr)
    norms_sq = np.sum(arr * arr, axis=1)

    # Greedy selection: pick vectors one by one, maintaining constraint
    # Constraint: for all pairs in selected, dist² >= max_norm_sq in selected
    selected = []
    
    # Try multiple orderings
    # First try: descending norm
    for order in [np.argsort(-norms_sq), np.argsort(norms_sq), np.random.RandomState(42).permutation(n)]:
        if time.time() > deadline - 10:
            break
        sel = []
        max_ns = 0
        for idx in order:
            if time.time() > deadline - 5:
                break
            v = arr[idx]
            ns = norms_sq[idx]
            new_max_ns = max(max_ns, ns)
            if not sel:
                sel.append(idx)
                max_ns = ns
                continue
            # Check distance to all selected
            sel_arr = arr[sel]
            diff = sel_arr - v
            dist_sq = np.sum(diff * diff, axis=1)
            if np.min(dist_sq) >= new_max_ns:
                sel.append(idx)
                max_ns = new_max_ns
        if len(sel) > len(selected):
            selected = sel

    if not selected:
        # Fallback: just type 1
        return arr[:220]

    result = arr[selected]
    return result
