"""Kissing configuration in R^11 via large sparse candidate pool + greedy selection."""
import time
import itertools
import numpy as np

D = 11


def construct():
    deadline = time.time() + 140.0  # safety margin vs 150s kill

    # ---- Build a large candidate pool of sparse integer vectors ----
    # Vectors with support s (1..5) and coefficients in {-2,-1,1,2}.
    # Support 5 provides a good balance of pool size and sparsity.
    coeffs = (-2, -1, 1, 2)
    pool = []
    for s in range(1, 6):
        for idx in itertools.combinations(range(D), s):
            for vals in itertools.product(coeffs, repeat=s):
                v = np.zeros(D, dtype=np.int64)
                v[list(idx)] = vals
                pool.append(v)
    pool = np.asarray(pool, dtype=np.int64)  # shape (N, 11)

    # Precompute per-vector squared norms.
    norms = (pool * pool).sum(axis=1)  # shape (N,)

    # ---- Greedy selection ----
    # Sort by ASCENDING norm to keep the max_norm threshold low for longer,
    # allowing more vectors to be packed before the threshold becomes restrictive.
    order = np.argsort(norms)

    selected = []
    max_norm = 0

    for idx in order:
        if time.time() >= deadline:
            break
        if not selected:
            selected.append(int(idx))
            max_norm = int(norms[idx])
            continue
        
        v = pool[idx]
        n = int(norms[idx])
        
        # Check compatibility with all selected vectors
        # Condition: ||v - u||^2 >= max(||v||^2, ||u||^2)
        # This is equivalent to: v . u <= min(||v||^2, ||u||^2) / 2
        
        if len(selected) < 1000:
            # For small sets, direct vectorization is fine
            sel_vecs = pool[selected]
            sel_norms = norms[selected]
            dots = sel_vecs @ v
            thresholds = np.minimum(sel_norms, n) / 2.0
            valid = np.all(dots <= thresholds)
        else:
            # For larger sets, we can still vectorize, but it might be memory heavy.
            # However, K is likely to stay moderate in this greedy pass.
            sel_vecs = pool[selected]
            sel_norms = norms[selected]
            dots = sel_vecs @ v
            thresholds = np.minimum(sel_norms, n) / 2.0
            valid = np.all(dots <= thresholds)

        if valid:
            selected.append(int(idx))
            if n > max_norm:
                max_norm = n

    if not selected:
        # Fallback: a single nonzero vector is always a valid configuration.
        selected = [int(order[0])]

    result = np.asarray(pool[selected], dtype=np.int64)

    # ---- Final safety validation (cheap) ----
    k = len(result)
    if k > 1:
        ns = (result * result).sum(axis=1)
        mm = int(ns.max())
        ok = True
        for a in range(k):
            d2 = ((result[a] - result) ** 2).sum(axis=1)
            if d2.min() < mm:
                ok = False
                break
        if not ok:
            # Should never trigger; fall back to a trivially valid single vector.
            result = result[:1]

    return result
