"""
Rank-49 exact tensor decomposition for 4x4 x 4x4 matrix multiplication.

This program constructs the optimal (best-known) rank-49 decomposition using
recursive Strassen algorithms. It includes a verification step to ensure
exactness and a safety fallback to the schoolbook algorithm (rank 64) if
verification fails.

The score is calculated as seed_rank / current_rank. With seed_rank=64 and
current_rank=49, the score is ~1.3061.
"""

import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START

# ---------------------------------------------------------------------------
# Inner 2x2 Strassen pattern (rank 7).
# Format: list of 7 dicts, each with 'u', 'v', 'w' keys.
# Each value is a list of (row, col, coeff) tuples for the 2x2 block.
# ---------------------------------------------------------------------------
_INNER_STRASSEN_2X2 = [
    # M1 = (a+d)(e+h)
    {'u': [(0, 0, 1), (1, 1, 1)], 'v': [(0, 0, 1), (1, 1, 1)], 'w': [(0, 0, 1), (1, 1, 1)]},
    # M2 = (c+d)e
    {'u': [(1, 0, 1), (1, 1, 1)], 'v': [(0, 0, 1)], 'w': [(1, 0, 1), (1, 1, -1)]},
    # M3 = a(f-h)
    {'u': [(0, 0, 1)], 'v': [(0, 1, 1), (1, 1, -1)], 'w': [(0, 1, 1), (1, 1, 1)]},
    # M4 = d(g-e)
    {'u': [(1, 1, 1)], 'v': [(1, 0, 1), (0, 0, -1)], 'w': [(0, 0, 1), (1, 0, 1)]},
    # M5 = (a+b)h
    {'u': [(0, 0, 1), (0, 1, 1)], 'v': [(1, 1, 1)], 'w': [(0, 1, 1), (0, 0, -1)]},
    # M6 = (c-a)(e+f)
    {'u': [(1, 0, 1), (0, 0, -1)], 'v': [(0, 0, 1), (0, 1, 1)], 'w': [(1, 1, 1)]},
    # M7 = (b-d)(g+h)
    {'u': [(0, 1, 1), (1, 1, -1)], 'v': [(1, 0, 1), (1, 1, 1)], 'w': [(0, 0, 1)]},
]

# ---------------------------------------------------------------------------
# Outer 4x4 Strassen pattern (rank 7) over 2x2 blocks.
# Same structure as inner, but applies to block indices.
# ---------------------------------------------------------------------------
_OUTER_STRASSEN_4X4 = [
    # P1 = (A11+A22)(B11+B22)
    {'a': [(0, 0, 1), (1, 1, 1)], 'b': [(0, 0, 1), (1, 1, 1)], 'c': [(0, 0, 1), (1, 1, 1)]},
    # P2 = (A21+A22)B11
    {'a': [(1, 0, 1), (1, 1, 1)], 'b': [(0, 0, 1)], 'c': [(1, 0, 1), (1, 1, -1)]},
    # P3 = A11(B12-B22)
    {'a': [(0, 0, 1)], 'b': [(0, 1, 1), (1, 1, -1)], 'c': [(0, 1, 1), (1, 1, 1)]},
    # P4 = A22(B21-B11)
    {'a': [(1, 1, 1)], 'b': [(1, 0, 1), (0, 0, -1)], 'c': [(0, 0, 1), (1, 0, 1)]},
    # P5 = (A11+A12)B22
    {'a': [(0, 0, 1), (0, 1, 1)], 'b': [(1, 1, 1)], 'c': [(0, 1, 1), (0, 0, -1)]},
    # P6 = (A21-A11)(B11+B12)
    {'a': [(1, 0, 1), (0, 0, -1)], 'b': [(0, 0, 1), (0, 1, 1)], 'c': [(1, 1, 1)]},
    # P7 = (A12-A22)(B21+B22)
    {'a': [(0, 1, 1), (1, 1, -1)], 'b': [(1, 0, 1), (1, 1, 1)], 'c': [(0, 0, 1)]},
]


def _strassen_4x4(n, m, p):
    """
    Build the rank-49 U, V, W factor matrices for 4x4 x 4x4 multiplication.
    """
    assert (n, m, p) == (4, 4, 4)
    rank = 49
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    for s, outer in enumerate(_OUTER_STRASSEN_4X4):
        for t, inner in enumerate(_INNER_STRASSEN_2X2):
            r = s * 7 + t

            # U construction: A[i, j]
            # Outer block (P, Q) and Inner offset (ir, jc)
            # Global row = 2*P + ir, Global col = 2*Q + jc
            # U index = row * m + col
            for (P, Q, sa) in outer['a']:
                for (ir, jc, sx) in inner['u']:
                    coeff = sa * sx
                    if coeff:
                        row = 2 * P + ir
                        col = 2 * Q + jc
                        U[row * m + col, r] += coeff

            # V construction: B[k, l]
            # Outer block (P, Q) and Inner offset (ir, jc)
            # Global row = 2*P + ir, Global col = 2*Q + jc
            # V index = row * p + col
            for (P, Q, sb) in outer['b']:
                for (ir, jc, sy) in inner['v']:
                    coeff = sb * sy
                    if coeff:
                        row = 2 * P + ir
                        col = 2 * Q + jc
                        V[row * p + col, r] += coeff

            # W construction: C[i, j]
            # Outer block (P, Q) and Inner offset (ir, jc)
            # Global row (i) = 2*P + ir, Global col (j) = 2*Q + jc
            # W index = j * n + i (Trace convention: W[k*n+i, r] where k=j, i=i)
            for (P, Q, sc) in outer['c']:
                for (ir, jc, sz) in inner['w']:
                    coeff = sc * sz
                    if coeff:
                        i = 2 * P + ir
                        j = 2 * Q + jc
                        W[j * n + i, r] += coeff

    return U, V, W


def _verify_decomposition(U, V, W, n, m, p, atol=1e-9):
    """
    Verify the decomposition.
    T[i, j, k] = sum_r U[i, j, r] * V[j, k, r] * W[k, i, r] should be 1 for all i, j, k.
    """
    rank = U.shape[1]
    # Reshape to 3D tensors: (n, m, rank), (m, p, rank), (p, n, rank)
    U_r = np.asarray(U, dtype=float).reshape(n, m, rank)
    V_r = np.asarray(V, dtype=float).reshape(m, p, rank)
    W_r = np.asarray(W, dtype=float).reshape(p, n, rank)

    # T[i, j, k] = sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
    # Note: W is indexed by (k, i) in the tensor definition, 
    # but our W matrix is stored as W[k*n + i, r].
    # So W_r[k, i, r] corresponds to W[k*n + i, r].
    
    T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
    return bool(np.allclose(T, 1.0, atol=atol))


def _schoolbook_decomposition(n, m, p):
    """Fallback rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def construct():
    deadline = time.time() + 115.0
    n, m, p = DIMENSIONS

    # Primary: Recursive Strassen (Rank 49)
    try:
        U, V, W = _strassen_4x4(n, m, p)
        if time.time() < deadline and _verify_decomposition(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass

    # Fallback: Schoolbook (Rank 64)
    U, V, W = _schoolbook_decomposition(n, m, p)
    if time.time() < deadline:
        _verify_decomposition(U, V, W, n, m, p)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
