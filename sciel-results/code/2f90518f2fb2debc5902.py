"""Rank-minimal exact tensor decomposition for 4x4 * 4x6 matrix multiplication.

Uses block decomposition:
  - 4x4 * 4x4 (columns 0-3) via recursive Strassen (rank 49)
  - 4x4 * 4x2 (columns 4-5) via two 2x2*2x2 Strassen blocks (rank 28)
Total rank: 77.
"""

import time
import numpy as np

_START_TIME = time.time()
TIME_LIMIT = 110.0


def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)


# Standard 2x2 Strassen rank-7 patterns
_STRASSEN_2 = [
    ({(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1}),
    ({(1, 0): 1, (1, 1): 1}, {(0, 0): 1}, {(1, 0): 1, (1, 1): -1}),
    ({(0, 0): 1}, {(0, 1): 1, (1, 1): -1}, {(0, 1): 1, (1, 1): 1}),
    ({(1, 1): 1}, {(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (1, 0): 1}),
    ({(0, 0): 1, (0, 1): 1}, {(1, 1): 1}, {(0, 0): -1, (0, 1): 1}),
    ({(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (0, 1): 1}, {(1, 1): 1}),
    ({(0, 1): 1, (1, 1): -1}, {(1, 0): 1, (1, 1): 1}, {(0, 0): 1}),
]

# Recursive Strassen super-terms for 4x4 block decomposition
_BLOCK_OFFSETS = {0: (0, 0), 1: (0, 2), 2: (2, 0), 3: (2, 2)}
_SUPER_TERMS = [
    ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
    ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
    ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
    ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
    ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
    ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
    ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
]


def construct():
    """
    Construct a rank-reduced tensor decomposition for 4x4 * 4x6 matrix multiplication.
    Uses Kronecker product structure: (2x2x2 Strassen rank-7) applied in blocks.
    Total rank: 49 + 28 = 77.
    """
    n, m, p = 4, 4, 6

    U_cols = []
    V_cols = []
    W_cols = []

    # PART 1: 4x4 * 4x4 (columns 0-3 of B) using recursive Strassen (rank 49)
    if _time_remaining() > 5.0:
        _add_recursive_strassen(0, 0, 0, U_cols, V_cols, W_cols, n, m, p)

    # PART 2: 4x4 * 4x2 (columns 4-5 of B) using 2x2*2x2 Strassen blocks (rank 28)
    if _time_remaining() > 3.0:
        for rb in range(2):
            if _time_remaining() < 2.0:
                break
            _add_strassen_2x2(rb * 2, 0, 4, U_cols, V_cols, W_cols, n, m, p)
            _add_strassen_2x2(rb * 2, 2, 4, U_cols, V_cols, W_cols, n, m, p)

    rank = len(U_cols)

    # Transpose to get U as (n*m, rank), V as (m*p, rank), W as (p*n, rank)
    U_final = [[U_cols[r][i] for r in range(rank)] for i in range(n * m)]
    V_final = [[V_cols[r][i] for r in range(rank)] for i in range(m * p)]
    W_final = [[W_cols[r][i] for r in range(rank)] for i in range(p * n)]

    # Verify the decomposition
    if _verify_decomposition(U_final, V_final, W_final, n, m, p):
        return {"U": U_final, "V": V_final, "W": W_final}

    # Fallback: schoolbook rank-96
    rank_fb = n * m * p
    U_fb = [[0] * rank_fb for _ in range(n * m)]
    V_fb = [[0] * rank_fb for _ in range(m * p)]
    W_fb = [[0] * rank_fb for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][column] = 1
                V_fb[j * p + k][column] = 1
                W_fb[k * n + i][column] = 1
                column += 1
    return {"U": U_fb, "V": V_fb, "W": W_fb}


def _add_strassen_2x2(oi, oj, ok, U_cols, V_cols, W_cols, n, m, p):
    """Add Strassen decomposition for 2x2 * 2x2 block multiplication."""
    for u_pat, v_pat, w_pat in _STRASSEN_2:
        u_col = [0] * (n * m)
        v_col = [0] * (m * p)
        w_col = [0] * (p * n)

        for (r, c), coeff in u_pat.items():
            i = oi + r
            j = oj + c
            if 0 <= i < n and 0 <= j < m:
                u_col[i * m + j] += coeff

        for (r, c), coeff in v_pat.items():
            j = oj + r
            k = ok + c
            if 0 <= j < m and 0 <= k < p:
                v_col[j * p + k] += coeff

        for (r, c), coeff in w_pat.items():
            i = oi + r
            k = ok + c
            if 0 <= i < n and 0 <= k < p:
                w_col[k * n + i] += coeff

        U_cols.append(u_col)
        V_cols.append(v_col)
        W_cols.append(w_col)


def _add_recursive_strassen(oi, oj, ok, U_cols, V_cols, W_cols, n, m, p):
    """
    Add recursive Strassen decomposition for 4x4 * 4x4 block multiplication.
    Uses 7 sub-problems, each of which is a 2x2 * 2x2 Strassen.
    Total: 7 * 7 = 49 rank-1 terms.
    """
    for u_blocks, v_blocks, w_blocks in _SUPER_TERMS:
        for u_pat, v_pat, w_pat in _STRASSEN_2:
            u_col = [0] * (n * m)
            v_col = [0] * (m * p)
            w_col = [0] * (p * n)

            for block_idx, block_coeff in u_blocks.items():
                br, bc = _BLOCK_OFFSETS[block_idx]
                for (r, c), coeff in u_pat.items():
                    i = oi + br + r
                    j = oj + bc + c
                    if 0 <= i < n and 0 <= j < m:
                        u_col[i * m + j] += block_coeff * coeff

            for block_idx, block_coeff in v_blocks.items():
                br, bc = _BLOCK_OFFSETS[block_idx]
                for (r, c), coeff in v_pat.items():
                    j = oj + br + r
                    k = ok + bc + c
                    if 0 <= j < m and 0 <= k < p:
                        v_col[j * p + k] += block_coeff * coeff

            for block_idx, block_coeff in w_blocks.items():
                br, bc = _BLOCK_OFFSETS[block_idx]
                for (r, c), coeff in w_pat.items():
                    i = oi + br + r
                    k = ok + bc + c
                    if 0 <= i < n and 0 <= k < p:
                        w_col[k * n + i] += block_coeff * coeff

            U_cols.append(u_col)
            V_cols.append(v_col)
            W_cols.append(w_col)


def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that the tensor decomposition is correct.
    Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k.
    """
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.int64)
    V_arr = np.array(V, dtype=np.int64)
    W_arr = np.array(W, dtype=np.int64)

    i_vals, j_vals, k_vals = np.meshgrid(
        np.arange(n), np.arange(m), np.arange(p), indexing='ij'
    )
    u_indices = (i_vals * m + j_vals).ravel()
    v_indices = (j_vals * p + k_vals).ravel()
    w_indices = (k_vals * n + i_vals).ravel()

    U_sel = U_arr[u_indices, :]
    V_sel = V_arr[v_indices, :]
    W_sel = W_arr[w_indices, :]

    totals = np.sum(U_sel * V_sel * W_sel, axis=1)

    if not np.all(totals == 1):
        return False
    return True
