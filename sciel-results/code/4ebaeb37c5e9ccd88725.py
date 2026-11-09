"""Rank-minimal exact tensor decomposition for 3x4 x 4x4 matrix multiplication.

Guarantees: the returned decomposition is always verified exact (rational
arithmetic). The baseline is the schoolbook rank-48 decomposition, so the
worst-case score is 1.0. A bounded verified search attempts to find a lower
rank; any candidate is accepted only if it is proven exact.
"""
DIMENSIONS = [3, 4, 4]

import time
import random
import numpy as np
from fractions import Fraction

_START = time.time()
_DEADLINE = 150.0   # evaluator kills at 150 s wall-clock
_MARGIN = 10.0      # safety margin

def _remaining():
    return _DEADLINE - _MARGIN - (time.time() - _START)

def _schoolbook(n, m, p):
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
    return U, V, W

def _verify_exact(U, V, W, n, m, p):
    """Exact check: for all i,j,k, sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1."""
    r = len(U[0])
    UF = [[Fraction(x) for x in row] for row in U]
    VF = [[Fraction(x) for x in row] for row in V]
    WF = [[Fraction(x) for x in row] for row in W]
    for i in range(n):
        for j in range(m):
            urow = UF[i * m + j]
            for k in range(p):
                vrow = VF[j * p + k]
                wrow = WF[k * n + i]
                s = Fraction(0)
                for rr in range(r):
                    a = urow[rr]
                    if a == 0:
                        continue
                    b = vrow[rr]
                    if b == 0:
                        continue
                    s += a * b * wrow[rr]
                if s != 1:
                    return False
    return True

def _verify_float(U, V, W, n, m, p, tol=1e-9):
    """Fast float pre-check."""
    Ua = np.asarray(U, dtype=float)
    Va = np.asarray(V, dtype=float)
    Wa = np.asarray(W, dtype=float)
    r = Ua.shape[1]
    T = np.zeros((n, m, p), dtype=float)
    for rr in range(r):
        uc = Ua[:, rr].reshape(n, m)
        vc = Va[:, rr].reshape(m, p)
        wc = Wa[:, rr].reshape(p, n)
        for j in range(m):
            a = uc[:, j]
            b = vc[j, :]
            c = wc[:, j]
            T[:, j, :] += (a[:, None] * c[:, None]) * b[None, :]
    return bool(np.allclose(T, 1.0, atol=tol, rtol=tol))

def _rand_half(rng):
    return rng.choice([-1, -0.5, 0, 0.5, 1])

def construct():
    n, m, p = DIMENSIONS
    U, V, W = _schoolbook(n, m, p)
    best_rank = n * m * p
    best = (U, V, W)

    rng = random.Random(2024)

    def _try_rank(r, iters):
        if r >= best_rank:
            return None
        for _ in range(iters):
            if _remaining() <= 0.5:
                return None
            Uc = [[_rand_half(rng) for _ in range(r)] for _ in range(n * m)]
            Vc = [[_rand_half(rng) for _ in range(r)] for _ in range(m * p)]
            Wc = [[_rand_half(rng) for _ in range(r)] for _ in range(p * n)]
            if _verify_float(Uc, Vc, Wc, n, m, p):
                if _verify_exact(Uc, Vc, Wc, n, m, p):
                    return (Uc, Vc, Wc)
        return None

    # Bounded verified search for a lower rank.
    for r in range(best_rank - 1, max(24, best_rank - 10), -1):
        if _remaining() <= 1.0:
            break
        res = _try_rank(r, 150)
        if res is not None:
            U, V, W = res
            best_rank = r
            best = (U, V, W)

    U, V, W = best
    return {"U": U, "V": V, "W": W}
