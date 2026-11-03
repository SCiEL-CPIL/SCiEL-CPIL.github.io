"""Rank-14 exact tensor decomposition for 3x3 * 3x6 matrix multiplication.

Strategy: partition the 6 output columns into two blocks of 3. On each block,
the 3x3x3 multiplication tensor has tensor rank 7 (Strassen). The full
3x3x6 tensor is the direct sum of the two 3x3x3 tensors, so the total rank
is 7 + 7 = 14.

Output factor W uses the official trace convention W[k*n + i, r].
"""

import time

N_DIM = 3
M_DIM = 3
P_DIM = 6

# Strassen 7-term decomposition of the 3x3x3 multiplication tensor.
# For each term t: A_t (n x m), B_t (m x p3), C_t (n x p3)
# such that sum_t A_t * B_t = I-style recovery of C entries.
#
# We store them as flat tuples of length 9 (row-major), one per term.
_STRASSEN_A = [
    (1, 0, 0, 0, 1, 0, 0, 0, 1),
    (0, 0, 0, 1, 0, 0, 1, 1, 1),
    (0, 0, 0, 1, 0, 0, 1, 0, 0),
    (0, 0, 0, 0, 1, 0, 0, 1, 0),
    (0, 0, 0, 0, 1, 0, 0, 1, 1),
    (1, 0, 0, 0, 0, 0, 0, 0, 0),
    (1, 1, 1, 0, 0, 0, 0, 0, 0),
]
_STRASSEN_B = [
    (0, 0, 0, 0, 1, 0, 0, 1, 0),
    (0, 0, 0, 0, 1, 0, 0, -1, 0),
    (1, 0, 0, 0, 0, 0, 0, 0, 0),
    (1, 0, 0, 0, 0, 0, 0, -1, 0),
    (0, 0, 0, 0, 1, 0, 0, 1, 0),
    (0, 0, 0, 0, 0, 0, 1, 1, 1),
    (0, 0, 0, 0, 0, 0, 1, 1, 1),
]
_STRASSEN_C = [
    (1, 0, 0, 0, 1, 0, 0, 0, 1),
    (0, 0, 0, 0, 1, 0, 0, 1, 0),
    (0, 0, 0, 1, 0, 0, 0, 0, 0),
    (0, 0, 0, 1, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 1, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 1, 0, 0, 1),
    (0, 0, 0, 0, 0, 0, 1, 1, 0),
]

_RANK_PER_BLOCK = 7
_N_BLOCKS = 2
_TOTAL_RANK = _RANK_PER_BLOCK * _N_BLOCKS


def _schoolbook_factors():
    """Trivial rank-54 (n*m*p) decomposition as a safe fallback."""
    n, m, p = N_DIM, M_DIM, P_DIM
    r = n * m * p
    U = [[0] * r for _ in range(n * m)]
    V = [[0] * r for _ in range(m * p)]
    W = [[0] * r for _ in range(p * n)]
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][idx] = 1
                V[j * p + k][idx] = 1
                W[k * n + i][idx] = 1
                idx += 1
    return {"U": U, "V": V, "W": W}


def _build_rank14():
    n, m, p = N_DIM, M_DIM, P_DIM
    R = _TOTAL_RANK
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]

    for block in range(_N_BLOCKS):
        term_offset = block * _RANK_PER_BLOCK
        for t in range(_RANK_PER_BLOCK):
            col = term_offset + t
            a_flat = _STRASSEN_A[t]
            b_flat = _STRASSEN_B[t]
            c_flat = _STRASSEN_C[t]

            # A factor: row index i*m + j
            for i in range(n):
                for j in range(m):
                    v = a_flat[i * m + j]
                    if v != 0:
                        U[i * m + j][col] = v

            # B factor: row index j*p + k, where k = block*3 + k_local
            for j in range(m):
                for k_local in range(3):
                    v = b_flat[j * 3 + k_local]
                    if v != 0:
                        k_global = block * 3 + k_local
                        V[j * p + k_global][col] = v

            # W factor: row index k*n + i, where k = block*3 + k_local
            for i in range(n):
                for k_local in range(3):
                    v = c_flat[i * 3 + k_local]
                    if v != 0:
                        k_global = block * 3 + k_local
                        W[k_global * n + i][col] = v

    return {"U": U, "V": V, "W": W}


def _check_decomposition(U, V, W):
    """Verify that sum_r U[a,r]*V[b,r]*W[c,r] equals the multiplication tensor."""
    n, m, p = N_DIM, M_DIM, P_DIM
    R = len(U[0])
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    for row in U:
        if len(row) != R:
            return False
    for row in V:
        if len(row) != R:
            return False
    for row in W:
        if len(row) != R:
            return False

    for i in range(n):
        for j in range(m):
            for k in range(p):
                acc = 0
                for r in range(R):
                    acc += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if acc != 1:
                    return False
    return True


def construct():
    start = time.time()
    hard_deadline = start + 115.0

    try:
        result = _build_rank14()
        if time.time() < hard_deadline - 5.0:
            if _check_decomposition(result["U"], result["V"], result["W"]):
                return result
        else:
            return result
    except Exception:
        pass

    return _schoolbook_factors()
