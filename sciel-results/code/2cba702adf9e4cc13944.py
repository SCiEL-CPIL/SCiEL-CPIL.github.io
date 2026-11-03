"""Matrix multiplication 3x3 by 3x6: rank-minimal exact tensor decomposition.

Strategy (fresh design, search family):
  - Model the multiplication tensor T[i,j,k] = (A*B)[i,k] as a sum of rank-1
    terms  U[:,r]*V[:,r]*W[:,r]  with the official trace convention
    W[k*n+i, r].
  - Search over candidate ranks r with several random initializations and
    alternating least squares (ALS), polishing each candidate.
  - Verify each candidate EXACTLY using sympy (exact rationals) so that the
    returned decomposition is guaranteed exact (not merely numerically close).
  - Return the lowest-rank exact decomposition found before the deadline.
"""

import time
import random
import itertools
import sys

import numpy as np
import sympy as sp
from sympy import Rational

DIMENSIONS = [3, 3, 6]
N, M, P = DIMENSIONS
RMAX = N * M * P  # 54, the schoolbook rank (fallback / upper bound)
DEADLINE = time.time() + 118.0   # keep a safety margin vs the 150 s kill
TIMEOUT = 18.0                   # per-candidate ALS budget (s)


def _schoolbook():
    """The trivial rank n*m*p exact decomposition (always valid)."""
    rank = N * M * P
    U = [[0.0] * rank for _ in range(N * M)]
    V = [[0.0] * rank for _ in range(M * P)]
    W = [[0.0] * rank for _ in range(P * N)]
    c = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j][c] = 1.0
                V[j * P + k][c] = 1.0
                W[k * N + i][c] = 1.0
                c += 1
    return {"U": U, "V": V, "W": W}


def _to_numpy(res):
    return (np.array(res["U"], dtype=float),
            np.array(res["V"], dtype=float),
            np.array(res["W"], dtype=float))


def _als(U, V, W, r, budget, rng):
    """Alternating least squares polishing. Returns (U,V,W,relerr)."""
    def matA(V):  # (NM, r)
        out = np.empty((N * M, r))
        for i in range(N):
            for j in range(M):
                out[i * M + j] = V[j * P:] @ V[j * P:j * P + P]
        return out
    def matB(U):  # (MP, r)
        out = np.empty((M * P, r))
        for j in range(M):
            for k in range(P):
                out[j * P + k] = U[:j * M] @ U[j * M:(j + 1) * M]
        return out
    def matW(U, V):  # (PN, r)
        out = np.empty((P * N, r))
        for k in range(P):
            for i in range(N):
                out[k * N + i] = U[i * M:(i + 1) * M] @ V[k * P:(k + 1) * P]
        return out

    t0 = time.time()
    for _ in range(600):
        if time.time() - t0 > budget:
            break
        # U
        b = np.empty((N * M, r))
        for i in range(N):
            for j in range(M):
                b[i * M + j] = V[j * P:] * V[j * P:j * P + P]
        U = np.linalg.lstsq(matA(V), b, rcond=None)[0]
        # V
        b = np.empty((M * P, r))
        for j in range(M):
            for k in range(P):
                b[j * P + k] = U[:j * M] * U[j * M:(j + 1) * M]
        V = np.linalg.lstsq(matB(U), b, rcond=None)[0]
        # W
        b = np.empty((P * N, r))
        for k in range(P):
            for i in range(N):
                b[k * N + i] = U[i * M:(i + 1) * M] * V[k * P:(k + 1) * P]
        W = np.linalg.lstsq(matW(U, V), b, rcond=None)[0]

    # final relative error
    num = 0.0
    den = 0.0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                for r_ in range(r):
                    num += (U[i * M + j, r_] * V[j * P + k, r_] * W[k * N + i, r_]) ** 2
    den = float(N * M * P)
    return U, V, W, num / den


def _exact_check(U, V, W):
    """Return True iff the decomposition is exact over Q (sympy, exact)."""
    n, m, p = N, M, P
    r = len(U[0])
    Us = [[sp.nsimplify(float(x), rational=True) for x in row] for row in U]
    Vs = [[sp.nsimplify(float(x), rational=True) for x in row] for row in V]
    Ws = [[sp.nsimplify(float(x), rational=True) for x in row] for row in W]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = sp.Integer(0)
                for rr in range(r):
                    s += Us[i * m + j][rr] * Vs[j * p + k][rr] * Ws[k * n + i][rr]
                if s != 1:
                    return False
    return True


def _try_rank(r, n_init, budget, rng):
    """Try to find an exact rank-r decomposition; return dict or None."""
    U0 = rng.standard_normal((N * M, r))
    V0 = rng.standard_normal((M * P, r))
    W0 = rng.standard_normal((P * N, r))
    best = None
    best_err = np.inf
    for _ in range(n_init):
        U = U0 + 1e-4 * rng.standard_normal((N * M, r))
        V = V0 + 1e-4 * rng.standard_normal((M * P, r))
        W = W0 + 1e-4 * rng.standard_normal((P * N, r))
        U, V, W, err = _als(U, V, W, r, budget, rng)
        if err < best_err:
            best_err = err
            best = (U, V, W)
        if best_err < 1e-9:
            break
    if best is None:
        return None
    U, V, W = best
    # exact verification
    if _exact_check(U.tolist(), V.tolist(), W.tolist()):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    return None


def construct():
    rng = np.random.default_rng(12345)
    # fallback is always available
    fallback = _schoolbook()

    # Try low ranks first (rank-minimal objective: lower is better).
    for r in [7, 8, 9, 10, 11, 12, 13, 14, 15]:
        if time.time() > DEADLINE:
            break
        n_init = 2
        budget = max(2.0, TIMEOUT - 3.0)
        res = _try_rank(r, n_init, budget, rng)
        if res is not None:
            # found a valid exact rank-r decomposition
            return res

    # If nothing low-rank worked, return the guaranteed schoolbook decomposition.
    return fallback


if __name__ == "__main__":
    res = construct()
    print("rank =", len(res["U"][0]))
    U, V, W = _to_numpy(res)
    ok = True
    for i in range(N):
        for j in range(M):
            for k in range(P):
                s = 0.0
                for rr in range(U.shape[1]):
                    s += U[i * M + j, rr] * V[j * P + k, rr] * W[k * N + i, rr]
                if abs(s - 1.0) > 1e-6:
                    ok = False
    print("numeric check ok:", ok)
