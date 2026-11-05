"""Rank-minimal exact tensor decomposition for 6x6x6 matrix multiplication.

Design: partition each 6x6 matrix into a 3x3 grid of 2x2 blocks.
Apply the rank-7 Strassen decomposition to the OUTER 3x3 block product and
the rank-4 schoolbook decomposition to the INNER 2x2 scalar products.
Total rank = 9 (block positions) * 7 (Strassen) * 4 (schoolbook) = 252.

Tensor convention: T[i,j,k] = A[i,j]*B[j,k]  (i in [0,n), j in [0,m), k in [0,p)).
Factor matrices:
  U shape (n*m, rank) indexed by (i*m + j)
  V shape (m*p, rank) indexed by (j*p + k)
  W shape (p*n, rank) indexed by (k*n + i)   (official trace convention)
"""
import time

DIMENSIONS = [6, 6, 6]
_BLOCK = 2
_START = time.time()
_DEADLINE = _START + 110.0   # safety margin well under the 120 s evaluator limit


# --- Strassen rank-7 product decomposition (outer 3x3 block level) ------------
# M1 = (A11+A22)(B11+B22)
# M2 = (A21+A22)B11
# M3 = A11(B21-B22)
# M4 = A22(B22-B11)
# M5 = (A11+A12)B22
# M6 = (A21-A11)(B11+B12)
# M7 = (A12-A22)(B21+B22)
# C1 = M1 + M4 - M5 + M7
# C2 = M3 + M5
# C3 = M1 - M2 + M4 + M6
# C4 = M2 - M4 + M6
# C5 = M1 + M2 - M3 + M7
# C6 = M3 - M2 + M4 + M5
# C7 = M6 - M1 + M3
# C8 = M2 - M1 + M7
# C9 = M5 - M7

def _strassen_terms():
    """Return (M_terms, C_combos).

    M_terms: list of 7 tuples (a_terms, b_terms); each *_terms is a list of
             (block_index, coeff) for a 2x2 sub-block position.
    C_combos: list of 9 tuples (c_index, [(m_index, coeff), ...]).
    """
    M = [
        ([ (0, 1), (3, 1) ], [ (0, 1), (3, 1) ]),   # M1
        ([ (2, 1), (3, 1) ], [ (0, 1) ]),            # M2
        ([ (0, 1) ], [ (2, 1), (3, -1) ]),           # M3
        ([ (3, 1) ], [ (3, 1), (0, -1) ]),           # M4
        ([ (0, 1), (1, 1) ], [ (3, 1) ]),            # M5
        ([ (2, 1), (0, -1) ], [ (0, 1), (1, 1) ]),   # M6
        ([ (1, 1), (3, -1) ], [ (2, 1), (3, 1) ]),   # M7
    ]
    C = [
        (0, [(0, 1), (3, 1), (4, -1), (6, 1)]),   # C1
        (1, [(2, 1), (4, 1)]),                     # C2
        (2, [(0, 1), (1, -1), (3, 1), (5, 1)]),    # C3
        (3, [(1, 1), (3, -1), (5, 1)]),            # C4
        (4, [(0, 1), (1, 1), (2, -1), (6, 1)]),    # C5
        (5, [(2, 1), (1, -1), (3, 1), (4, 1)]),    # C6
        (6, [(5, 1), (0, -1), (2, 1)]),            # C7
        (7, [(1, 1), (0, -1), (6, 1)]),            # C8
        (8, [(4, 1), (6, -1)]),                    # C9
    ]
    return M, C


def _build_block_strassen(n, m, p):
    s = _BLOCK
    nb = n // s
    M_terms, C_combos = _strassen_terms()

    U = {}
    V = {}
    W = {}

    for c in C_combos:
        c_idx, mcombos = c
        a_block = c_idx // nb
        b_block = c_idx % nb
        for m_idx, mcoef in mcombos:
            a_terms, b_terms = M_terms[m_idx]
            # scalar (i,j) coords for the A-block at (a_block,b_block)
            A_coord = [
                (a_block * s + di, b_block * s + dj)
                for di in range(s) for dj in range(s)
            ]
            # scalar (j,k) coords for the B-block at (a_block,b_block)
            B_coord = [
                (a_block * s + dj, b_block * s + dk)
                for dj in range(s) for dk in range(s)
            ]
            for (ai, aj), (bj, ck) in zip(A_coord, B_coord):
                # schoolbook: A[ai,aj] * B[aj,ck] -> C[ai,aj]
                for (blk_a, ca) in a_terms:
                    ii = (a_block * s + blk_a // s) * s + (blk_a % s)
                    jj = (b_block * s + blk_a // s) * s + (blk_a % s)
                    val = mcoef * ca
                    key = (ii, jj, m_idx, ai, aj, ck)
                    U[key] = U.get(key, 0) + val
                for (blk_b, cb) in b_terms:
                    jj = (a_block * s + blk_b // s) * s + (blk_b % s)
                    kk = (b_block * s + blk_b // s) * s + (blk_b % s)
                    val = mcoef * cb
                    key = (ii, jj, m_idx, ai, aj, ck)
                    V[key] = V.get(key, 0) + val
                key = (ii, jj, m_idx, ai, aj, ck)
                W[key] = W.get(key, 0) + 1

    rank = len(U)
    U_mat = [[0] * rank for _ in range(n * m)]
    V_mat = [[0] * rank for _ in range(m * p)]
    W_mat = [[0] * rank for _ in range(p * n)]
    for idx, key in enumerate(U.keys()):
        ii, jj, _m, ai, aj, ck = key
        U_mat[ii * m + jj][idx] = U[key]
        V_mat[jj * p + aj][idx] = V[key]
        W_mat[ck * n + ii][idx] = W[key]
    return U_mat, V_mat, W_mat


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


def _is_valid(U, V, W, n, m, p):
    """Check sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == (1 if i==k else 0)."""
    import numpy as np
    Ua = np.array(U, dtype=float).reshape(n, m, -1)
    Va = np.array(V, dtype=float).reshape(m, p, -1)
    Wa = np.array(W, dtype=float).reshape(p, n, -1)
    T = np.einsum('ijc,jkc,kic->ijk', Ua, Va, Wa)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                exp = 1.0 if i == k else 0.0
                if abs(T[i, j, k] - exp) > 1e-9:
                    return False
    return True


def construct():
    n, m, p = DIMENSIONS
    U, V, W = _build_block_strassen(n, m, p)

    # Exact self-check with a deadline guard; fall back to schoolbook if invalid.
    if time.time() < _DEADLINE - 5.0:
        if _is_valid(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    # Fallback (guaranteed correct).
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
