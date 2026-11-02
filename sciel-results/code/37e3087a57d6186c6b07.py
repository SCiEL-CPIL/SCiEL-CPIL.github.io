"""Packing integer vectors in R^11 for kissing-number lower bound."""
import time
import itertools
import numpy as np

D = 11


def _gen_weight(w):
    """All vectors in {-1,0,1}^D with exactly w nonzeros."""
    vecs = []
    for pos in itertools.combinations(range(D), w):
        for signs in itertools.product((1, -1), repeat=w):
            v = [0] * D
            for p, s in zip(pos, signs):
                v[p] = s
            vecs.append(v)
    return np.array(vecs, dtype=np.int8)


def _greedy(order, C, n):
    """Greedy maximal independent set given a conflict matrix C and an order."""
    selected = np.zeros(n, dtype=bool)
    blocked = np.zeros(n, dtype=bool)
    k = 0
    for idx in order:
        if selected[idx] or blocked[idx]:
            continue
        selected[idx] = True
        k += 1
        blocked |= C[idx]
    return selected, k


def _best_for_weight(w, deadline, rng):
    V = _gen_weight(w)
    n = V.shape[0]
    dot = V @ V.T
    T = w // 2
    C = dot > T
    static_deg = C.sum(axis=1)

    # Static-degree ordering first.
    order = np.argsort(static_deg)
    sel, k = _greedy(order, C, n)
    best_sel, best_k = sel.copy(), k

    # Weight-2 is trivially complete; no restarts needed.
    if w == 2:
        return best_sel, best_k, V

    # Random restarts until the time slice is exhausted.
    while time.time() < deadline:
        order = rng.permutation(n)
        sel, k = _greedy(order, C, n)
        if k > best_k:
            best_k = k
            best_sel = sel.copy()
    return best_sel, best_k, V


def construct():
    deadline = time.time() + 140.0
    rng = np.random.default_rng(12345)

    best_sel = None
    best_k = 0
    best_V = None

    for w in (2, 3, 4):
        if time.time() > deadline:
            break
        slice_seconds = 2.0 if w == 2 else 35.0
        slice_deadline = min(deadline - 3.0, time.time() + slice_seconds)
        sel, k, V = _best_for_weight(w, slice_deadline, rng)
        if sel is not None and k > best_k:
            best_k = k
            best_sel = sel
            best_V = V

    if best_sel is None or best_k == 0:
        return _gen_weight(2).astype(np.int64)

    out = best_V[best_sel].astype(np.int64)

    # Cheap validity guard: equal norm M=√w, need min pairwise dist >= M.
    if len(out) > 1:
        g = out.astype(np.float64) @ out.astype(np.float64).T
        w = best_V[best_sel].astype(np.float64)
        M2 = (w ** 2).sum(axis=1)[0]
        dist2 = 2.0 * M2 - 2.0 * g
        np.fill_diagonal(dist2, np.inf)
        if dist2.min() < M2 - 1e-6:
            return _gen_weight(2).astype(np.int64)

    return out
