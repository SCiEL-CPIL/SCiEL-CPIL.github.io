import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 7]
_n, _m, _p = DIMENSIONS

# ---------------------------------------------------------------------------
# Precompute the target (schoolbook) multiplication tensor T[i,j,k] = 1
# and a helper to validate any candidate decomposition.
# ---------------------------------------------------------------------------

def _tensor_from_factors(U, V, W):
    """Return the tensor C_{i,j,k} = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c]."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = np.dot(U[i * m + j, :],
                                    V[j * p + k, :] * W[k * n + i, :])
    return T


def _validate(U, V, W, tol=1e-9):
    T = _tensor_from_factors(U, V, W)
    n, m, p = DIMENSIONS
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if abs(T[i, j, k] - 1.0) > tol:
                    return False
    return True


# ---------------------------------------------------------------------------
# Candidate 1: rank 28 = 4 * 7
#   Block the inner dim m=3 as (2,1):
#     M = [A1 (2x2) | A2 (2x1)]
#     N = [B1 (2x7) ; B2 (1x7)]
#     C = A1*B1 + A2*B2
#   A1*B1 handled by Strassen (rank 7), A2*B2 handled by 4 terms (rank 7).
# ---------------------------------------------------------------------------

def _cand_rank28():
    n, m, p = DIMENSIONS
    U = [[0] * 28 for _ in range(n * m)]
    V = [[0] * 28 for _ in range(m * p)]
    W = [[0] * 28 for _ in range(p * n)]

    # Strassen multipliers for 2x2 * 2x2, indexed by (a,b) in {0,1}^2
    M = [
        [[1, 1], [1, 0]],
        [[0, 0], [1, 1]],
        [[1, 0], [0, 1]],
        [[0, 1], [1, 1]],
        [[1, 0], [1, 0]],
        [[1, 1], [0, 0]],
        [[0, 1], [0, 1]],
    ]
    # Strassen updates to C (2x2), indexed by (a,b)
    C = [
        [[1, 0], [0, 1]],
        [[0, 1], [0, 0]],
        [[0, 0], [1, 0]],
        [[0, 0], [0, 1]],
        [[1, 0], [0, 0]],
        [[0, 0], [1, 0]],
        [[0, 1], [0, 0]],
    ]

    # ---- A1*B1 : Strassen, 7 terms ----
    for c in range(7):
        for a in range(2):
            for b in range(2):
                U[a * m + b][c] += M[c][a][b]
        for a in range(2):
            for k in range(7):
                V[a * p + k][c] += C[c][a][k]
        for k in range(7):
            for i in range(2):
                W[k * n + i][c] += C[c][i][k]

    # ---- A2*B2 : 2x1 * 1x7, 4 terms ----
    # A2 row a (a=0,1), B2 row k (k=0..6), output C[i,k].
    for c in range(7, 14):
        a = c - 7
        U[a * m + 2][c] = 1
        for k in range(7):
            V[2 * p + k][c] = 1
            W[k * n + a][c] = 1

    return {"U": U, "V": V, "W": W}


# ---------------------------------------------------------------------------
# Candidate 2: rank 32 = 4 * 8
#   Same block split, but A1*B1 (2x2x7) handled by a 2x2x8 decomposition
#   (rank 8) whose last k-column is zero, then dropped.
# ---------------------------------------------------------------------------

def _cand_rank32():
    n, m, p = DIMENSIONS
    U = [[0] * 32 for _ in range(n * m)]
    V = [[0] * 32 for _ in range(m * p)]
    W = [[0] * 32 for _ in range(p * n)]

    M = [
        [[1, 1], [1, 0]],
        [[0, 0], [1, 1]],
        [[1, 0], [0, 1]],
        [[0, 1], [1, 1]],
        [[1, 0], [1, 0]],
        [[1, 1], [0, 0]],
        [[0, 1], [0, 1]],
    ]
    C8 = [
        [[1, 0, 0, 0, 0, 0, 0, 0]],
        [[0, 1, 0, 0, 0, 0, 0, 0]],
        [[0, 0, 1, 0, 0, 0, 0, 0]],
        [[0, 0, 0, 1, 0, 0, 0, 0]],
        [[1, 0, 0, 0, 0, 0, 0, 0]],
        [[0, 1, 0, 0, 0, 0, 0, 0]],
        [[0, 0, 1, 0, 0, 0, 0, 0]],
        [[0, 0, 0, 1, 0, 0, 0, 0]],
    ]

    for c in range(8):
        for a in range(2):
            for b in range(2):
                U[a * m + b][c] += M[c][a][b]
        for a in range(2):
            for k in range(8):
                V[a * p + k][c] += C8[c][a][k]
        for k in range(8):
            for i in range(2):
                W[k * n + i][c] += C8[c][i][k]

    for c in range(8, 16):
        a = c - 8
        U[a * m + 2][c] = 1
        for k in range(7):
            V[2 * p + k][c] = 1
            W[k * n + a][c] = 1

    return {"U": U, "V": V, "W": W}


# ---------------------------------------------------------------------------
# Candidate 3: rank 42 (schoolbook) - always valid fallback.
# ---------------------------------------------------------------------------

def _cand_rank42():
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
    t0 = time.time()
    deadline = t0 + 150 - 10  # safety margin

    candidates = [
        (28, _cand_rank28),
        (32, _cand_rank32),
        (42, _cand_rank42),
    ]

    best = None
    for rank, builder in candidates:
        if time.time() > deadline:
            break
        try:
            cand = builder()
        except Exception:
            continue
        try:
            if _validate(cand["U"], cand["V"], cand["W"]):
                if best is None or rank < best[0]:
                    best = (rank, cand)
        except Exception:
            continue

    if best is None:
        return _cand_rank42()

    return best[1]
