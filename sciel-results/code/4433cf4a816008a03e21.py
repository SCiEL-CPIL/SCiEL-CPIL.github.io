"""Search for a rank-minimal exact tensor decomposition of 3x5 * 5x6 multiplication."""
import time
import itertools
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
_DEADLINE = time.time() + 110.0   # hard wall-clock guard (evaluator kills at 150s)

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p):
    """Check sum_r U[a,r] V[b,r] == (a*b % m)*(a*b // m) for all a in [0,nm), b in [0,mp)."""
    U = np.asarray(U, dtype=object)
    V = np.asarray(V, dtype=object)
    r = U.shape[1]
    nm = n * m
    mp = m * p
    # U @ V.T -> (nm, mp)
    prod = U @ V.T
    for a in range(nm):
        i = a // m
        j = a % m
        for b in range(mp):
            k = b // p
            ell = b % p
            if prod[a, b] != (j * k):
                return False
    return True

def _verify_W(U, V, W, n, m, p):
    r = len(U[0])
    # build expected tensor T[a,b,c]
    nm = n * m; mp = m * p; pn = p * n
    # For each r, outer U[:,r] (nm) x V[:,r] (mp) must match contribution to W
    # Check: sum_r U[a,r]*V[b,r] == (a*b%m)*(a*b//m)  (same as _verify) AND
    # consistency: T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r] must equal delta.
    U = np.asarray(U, dtype=object); V = np.asarray(V, dtype=object); W = np.asarray(W, dtype=object)
    for a in range(nm):
        i = a // m; j = a % m
        for b in range(mp):
            k = b // p; ell = b % p
            s = 0
            for rr in range(r):
                uv = U[a, rr] * V[b, rr]
                if uv != 0:
                    s += uv * W[c_index(ell, i), rr]
            if s != (j * k):
                return False
    return True

def c_index(ell, i):
    # W index = k*n + i  (trace convention)
    return ell * (3) + i  # n=3

def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    full = n * m * p
    best = _schoolbook(n, m, p)

    def timed():
        return time.time() > _DEADLINE

    # Deterministic integer search for a rank strictly below full.
    # We look for a small-rank decomposition using the structure of the tensor.
    # Strategy: try ranks from (full-1) downward is far too large; instead attempt
    # a structured reduction via combining columns (Strassen-like) is infeasible
    # for 3x5x6 without heavy search. So we do a bounded random integer search.
    import random
    rng = random.Random(12345)
    for rank in range(full - 1, 0, -1):
        if timed():
            break
        if rank < 30:  # don't burn time on tiny ranks with low success
            break
        attempts = 0
        max_attempts = 40
        while attempts < max_attempts:
            attempts += 1
            if timed():
                break
            U = [[0] * rank for _ in range(nm)]
            V = [[0] * rank for _ in range(mp)]
            W = [[0] * rank for _ in range(pn)]
            for rr in range(rank):
                for a in range(nm):
                    U[a][rr] = rng.choice([-1, 0, 1, 1])
                for b in range(mp):
                    V[b][rr] = rng.choice([-1, 0, 1, 1])
                for c in range(pn):
                    W[c][rr] = rng.choice([-1, 0, 1, 1])
            if _verify(U, V, W, n, m, p):
                best = {"U": U, "V": V, "W": W}
                break
        if best is not _schoolbook(n, m, p):
            break
    return best
# EVOLVE-BLOCK-END
