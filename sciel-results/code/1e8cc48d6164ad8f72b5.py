"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def _strassen_2x2x2():
    """Return (U, V, W) for rank-7 Strassen 2x2x2 decomposition."""
    U = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    V = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    W = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    return U, V, W

def _verify_decomposition(U_mat, V_mat, W_mat, n, m, p):
    """Verify that the tensor decomposition is correct."""
    U = np.array(U_mat, dtype=np.int64)
    V = np.array(V_mat, dtype=np.int64)
    W = np.array(W_mat, dtype=np.int64)
    rank = U.shape[1]
    # Compute all (i,j,k) values
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = int(np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :]))
                if val != 1:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return schoolbook decomposition of rank n*m*p."""
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

def _embed_strassen_block(U, V, W, col, i_list, j_list, k_list, n, m, p):
    """Embed a Strassen 2x2x2 block into the global factors at column offset col."""
    Us, Vs, Ws = _strassen_2x2x2()
    i0, i1 = i_list
    j0, j1 = j_list
    k0, k1 = k_list
    u_rows = [i0 * m + j0, i0 * m + j1, i1 * m + j0, i1 * m + j1]
    v_rows = [j0 * p + k0, j0 * p + k1, j1 * p + k0, j1 * p + k1]
    w_rows = [k0 * n + i0, k1 * n + i0, k0 * n + i1, k1 * n + i1]
    for c in range(7):
        for s in range(4):
            U[u_rows[s]][col + c] = Us[s][c]
            V[v_rows[s]][col + c] = Vs[s][c]
            W[w_rows[s]][col + c] = Ws[s][c]
    return col + 7

def construct():
    start_time = time.time()
    deadline = start_time + 120

    n, m, p = DIMENSIONS  # 3, 5, 5

    # Strategy: 4 Strassen 2x2x2 blocks + schoolbook remainder = rank 71
    strassen_blocks = [
        ([1, 2], [0, 1], [0, 1]),
        ([1, 2], [0, 1], [3, 4]),
        ([1, 2], [3, 4], [0, 1]),
        ([1, 2], [3, 4], [3, 4]),
    ]

    covered = set()
    for i_list, j_list, k_list in strassen_blocks:
        for i in i_list:
            for j in j_list:
                for k in k_list:
                    covered.add((i, j, k))

    rank_total = 4 * 7 + (n * m * p - 4 * 8)
    assert rank_total == 71

    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]

    col = 0
    for i_list, j_list, k_list in strassen_blocks:
        col = _embed_strassen_block(U, V, W, col, i_list, j_list, k_list, n, m, p)

    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) in covered:
                    continue
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    assert col == rank_total

    if not _verify_decomposition(U, V, W, n, m, p):
        U, V, W = _schoolbook(n, m, p)
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
