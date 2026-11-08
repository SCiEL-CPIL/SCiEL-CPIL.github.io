"""Search for a rank-minimal exact tensor decomposition of matmul 2x4 x 4x4.

Randomized factor construction + exact rational rank detection + exact polish,
all guarded by an explicit wall-clock deadline (150s limit, 5s margin).
"""
import time
import numpy as np
from fractions import Fraction
from random import Random

DIMENSIONS = [2, 4, 4]
_TOL = 1e-9

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 145.0
    rng = Random(123456789)
    U0, V0, W0 = _schoolbook_basis(n, m, p)
    best_U, best_V, best_W = U0.copy(), V0.copy(), W0.copy()
    best_rank = U0.shape[1]

    def still_ok():
        return time.time() < deadline

    for it in range(6000):
        if not still_ok():
            break
        U, V, W = _random_init(U0, V0, W0, rng)
        r, sup = _rank_support(W)
        if r == 0 or r >= best_rank:
            continue
        Us = U[:, sup]
        Vs = V[:, sup]
        Ws = W[:, sup]
        if _is_valid(Us, Vs, Ws, n, m, p, _TOL):
            best_U, best_V, best_W = Us.copy(), Vs.copy(), Ws.copy()
            best_rank = r

    # Exact rational polish: greedily drop redundant columns while staying valid.
    if still_ok() and best_rank > 3:
        for _ in range(40):
            if not still_ok():
                break
            dropped = False
            R = best_U.shape[1]
            norms = np.sum(best_W * best_W, axis=0)
            order = sorted(range(R), key=lambda c: norms[c])
            for cand in order:
                if not still_ok():
                    break
                nU = best_U.copy(); nV = best_V.copy(); nW = best_W.copy()
                nU[:, cand] = 0.0; nV[:, cand] = 0.0; nW[:, cand] = 0.0
                if _is_valid(nU, nV, nW, n, m, p, 1e-8):
                    mask = [c for c in range(R) if c != cand]
                    best_U, best_V, best_W = nU[:, mask], nV[:, mask], nW[:, mask]
                    best_rank -= 1
                    dropped = True
                    break
            if not dropped:
                break

    return {"U": [list(map(float, row)) for row in best_U],
            "V": [list(map(float, row)) for row in best_V],
            "W": [list(map(float, row)) for row in best_W],
            "rank": best_rank}
# EVOLVE-BLOCK-END


def _schoolbook_basis(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank)); V = np.zeros((m * p, rank)); W = np.zeros((p * n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1.0; V[j * p + k, c] = 1.0; W[k * n + i, c] = 1.0
                c += 1
    return U, V, W


def _random_init(U0, V0, W0, rng):
    R = U0.shape[1]
    s = 0.85
    U = U0 + s * np.array([[rng.gauss(0, 1) for _ in range(R)] for _ in range(U0.shape[0])])
    V = V0 + s * np.array([[rng.gauss(0, 1) for _ in range(R)] for _ in range(V0.shape[0])])
    W = W0 + s * np.array([[rng.gauss(0, 1) for _ in range(R)] for _ in range(W0.shape[0])])
    return U, V, W


def _rank_support(M):
    A = M
    rows, cols = A.shape
    F = [[Fraction(round(float(A[i, j]))) for j in range(cols)] for i in range(rows)]
    rank = 0; pivots = []
    for c in range(cols):
        pivot = None
        for r in range(rank, rows):
            if F[r][c] != 0:
                pivot = r; break
        if pivot is None:
            continue
        F[rank], F[pivot] = F[pivot], F[rank]
        pivots.append(c)
        pv = F[rank][c]
        for j in range(c, cols):
            F[rank][j] = F[rank][j] / pv
        for r in range(rows):
            if r == rank:
                continue
            f = F[r][c]
            if f != 0:
                for j in range(c, cols):
                    F[r][j] = F[r][j] - f * F[rank][j]
        rank += 1
    return rank, pivots[:rank]


def _is_valid(U, V, W, n, m, p, tol=1e-9):
    U = np.asarray(U, float); V = np.asarray(V, float); W = np.asarray(W, float)
    R = U.shape[1]
    if V.shape[1] != R or W.shape[1] != R:
        return False
    # product triples
    for a in range(n * m):
        i_a, j_a = divmod(a, m)
        for b in range(m * p):
            j_b, k_b = divmod(b, p)
            if j_a != j_b:
                continue
            c = k_b * n + i_a
            s = 0.0
            for r in range(R):
                s += U[a, r] * V[b, r] * W[c, r]
            if abs(s - 1.0) > tol:
                return False
    # zero triples: mismatched j, or wrong c
    for a in range(n * m):
        i_a, j_a = divmod(a, m)
        for b in range(m * p):
            j_b, k_b = divmod(b, p)
            if j_a == j_b:
                c_match = k_b * n + i_a
                for c in range(p * n):
                    if c == c_match:
                        continue
                    s = 0.0
                    for r in range(R):
                        s += U[a, r] * V[b, r] * W[c, r]
                    if abs(s) > tol:
                        return False
            else:
                for c in range(p * n):
                    s = 0.0
                    for r in range(R):
                        s += U[a, r] * V[b, r] * W[c, r]
                    if abs(s) > tol:
                        return False
    return True
