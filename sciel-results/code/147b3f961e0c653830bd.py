"""Rewritten: structured low-rank exact tensor decomposition search for 3x4x7 MM."""
import time
import random
import numpy as np

DIMENSIONS = [3, 4, 7]

_DEADLINE = None
_DEADLINE_MARGIN = 1.0


def _target_tensor():
    n, m, p = DIMENSIONS
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    return T


def _pattern_set():
    """Structured rank-1 patterns: (i + a*j + b*k) mod n for a in Z_m, b in Z_p."""
    n, m, p = DIMENSIONS
    pats = []
    for a in range(m):
        for b in range(p):
            v = np.zeros((n, m, p))
            for j in range(m):
                for k in range(p):
                    i = (a * j + b * k) % n
                    v[i, j, k] = 1.0
            pats.append(v)
    return pats


def _solve_exact(patterns, t_flat, max_rank, deadline):
    """Find smallest r such that t_flat is in span of patterns[:r] exactly."""
    t = t_flat.astype(np.float64)
    best = None
    for r in range(1, max_rank + 1):
        if _elapsed() > deadline:
            break
        P = np.stack([patterns[c].flatten() for c in range(r)], axis=1)
        # exactness: residual norm of projection onto col(P)
        try:
            sol, res, rank, sv = np.linalg.lstsq(P, t, rcond=None)
        except np.linalg.LinAlgError:
            continue
        resid = float(np.linalg.norm(P @ sol - t))
        if resid < 1e-6:
            best = (r, sol)
            break
    return best


def _elapsed():
    return time.time() - _DEADLINE


def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def construct():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time()
    deadline = _DEADLINE + 110.0  # safety margin under 120s limit

    n, m, p = DIMENSIONS
    T = _target_tensor()
    t_flat = T.reshape(-1)

    # Build structured pattern pool (cyclic group algebra style).
    pats = _pattern_set()
    rng = random.Random(12345)
    idx = list(range(len(pats)))
    rng.shuffle(idx)

    # Try to find a low-rank exact combination.
    found = None
    max_rank = 40
    for r in range(1, max_rank + 1):
        if time.time() > deadline:
            break
        # pick r patterns (first r of shuffled order)
        cols = idx[:r]
        P = np.stack([pats[c].flatten() for c in cols], axis=1)
        try:
            sol, res, rank, sv = np.linalg.lstsq(P, t_flat, rcond=None)
        except np.linalg.LinAlgError:
            continue
        resid = float(np.linalg.norm(P @ sol - t))
        if resid < 1e-6:
            found = (r, cols, sol)
            break

    if found is not None:
        r, cols, sol = found
        # Build U,V,W from coefficients.
        U = [[0.0] * r for _ in range(n * m)]
        V = [[0.0] * r for _ in range(m * p)]
        W = [[0.0] * r for _ in range(p * n)]
        for c, pat_idx in enumerate(cols):
            coeff = float(sol[c])
            if abs(coeff) < 1e-12:
                continue
            v = pats[pat_idx]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if v[i, j, k] != 0:
                            U[i * m + j][c] += coeff * 1.0
                            V[j * p + k][c] += coeff * 1.0
                            W[k * n + i][c] += coeff * 1.0
        # Round to nearest half-integer for clean output.
        def rnd(x):
            return round(x * 2) / 2.0
        U = [[rnd(x) for x in row] for row in U]
        V = [[rnd(x) for x in row] for row in V]
        W = [[rnd(x) for x in row] for row in W]
        return {"U": U, "V": V, "W": W}

    # Fallback: schoolbook rank-84.
    return _schoolbook()
