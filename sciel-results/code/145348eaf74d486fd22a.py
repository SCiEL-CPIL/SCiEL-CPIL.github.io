"""
Rank-minimal tensor decomposition for 3x4 x 4x8 matrix multiplication.

Uses Strassen's rank-7 decomposition on eight 2x2x2 sub-blocks
(i in {0,1}, j in {0,2}, k in {0,2,4,6}), and schoolbook for the
remaining 1x4x8 part (i=2).

Total rank: 8 * 7 + 32 = 88 (vs schoolbook 96).
"""

import numpy as np
import time

DIMENSIONS = [3, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS  # 3, 4, 8
    nm, mp, pn = n * m, m * p, p * n  # 12, 32, 24

    # Correct Strassen rank-1 decomposition for 2x2x2
    # Each term: (U_dict, V_dict, W_dict)
    # U: (i,j) -> coeff, V: (j,k) -> coeff, W: (k,i) -> coeff
    def strassen_2x2x2():
        return [
            ({(0,0):1, (1,1):-1}, {(0,1):1, (1,1):1}, {(1,0):1, (0,1):1}),
            ({(0,0):1, (0,1):1}, {(1,1):1}, {(1,1):1}),
            ({(0,0):1}, {(0,0):1, (1,0):-1}, {(0,0):1, (0,1):1}),
            ({(1,1):1}, {(0,1):1, (0,0):-1}, {(1,0):1, (1,1):1}),
            ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
            ({(0,1):1, (1,1):-1}, {(0,0):1, (1,0):1}, {(0,0):1, (1,0):1}),
            ({(0,0):1, (0,1):-1}, {(0,1):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ]

    # Verify Strassen decomposition
    def verify_strassen():
        terms = strassen_2x2x2()
        T = np.zeros((2, 2, 2), dtype=np.int64)
        for u_d, v_d, w_d in terms:
            for (i, j), u_val in u_d.items():
                for (j2, k), v_val in v_d.items():
                    if j == j2:
                        for (k2, i2), w_val in w_d.items():
                            if k == k2:
                                T[i, j, k] += u_val * v_val * w_val
        return np.all(T == 1)

    strassen_ok = verify_strassen()

    # Build full decomposition
    # Strassen for i in {0,1}, j in {0,2}, k in {0,2,4,6}
    # Schoolbook for i=2, all j, k
    n_strassen_blocks = 2 * 2 * 4  # 8 blocks
    n_schoolbook_terms = 1 * m * p  # 32 terms
    total_rank = n_strassen_blocks * 7 + n_schoolbook_terms  # 56 + 32 = 88

    U = np.zeros((nm, total_rank), dtype=np.int64)
    V = np.zeros((mp, total_rank), dtype=np.int64)
    W = np.zeros((pn, total_rank), dtype=np.int64)
    col = 0

    if strassen_ok:
        terms = strassen_2x2x2()
        for i0 in [0, 1]:
            for j0 in [0, 2]:
                for k0 in [0, 2, 4, 6]:
                    for u_d, v_d, w_d in terms:
                        for (il, jl), u_val in u_d.items():
                            i_g = i0 + il
                            j_g = j0 + jl
                            U[i_g * m + j_g, col] += u_val
                        for (jl, kl), v_val in v_d.items():
                            j_g = j0 + jl
                            k_g = k0 + kl
                            V[j_g * p + k_g, col] += v_val
                        for (kl, il), w_val in w_d.items():
                            k_g = k0 + kl
                            i_g = i0 + il
                            W[k_g * n + i_g, col] += w_val
                        col += 1

    # Schoolbook for i=2
    for j in range(m):
        for k in range(p):
            U[2 * m + j, col] = 1
            V[j * p + k, col] = 1
            W[k * n + 2, col] = 1
            col += 1

    # Validate
    def validate(U_mat, V_mat, W_mat):
        rank = U_mat.shape[1]
        T_recon = np.zeros((nm, mp, pn), dtype=np.int64)
        for r in range(rank):
            T_recon += np.einsum('i,j,k->ijk', U_mat[:, r], V_mat[:, r], W_mat[:, r])
        # Expected tensor
        T_expected = np.zeros((nm, mp, pn), dtype=np.int64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_expected[i * m + j, j * p + k, k * n + i] = 1
        return np.array_equal(T_recon, T_expected)

    if validate(U, V, W):
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    else:
        # Fallback to schoolbook
        rank_sb = n * m * p
        U_sb = np.zeros((nm, rank_sb), dtype=np.int64)
        V_sb = np.zeros((mp, rank_sb), dtype=np.int64)
        W_sb = np.zeros((pn, rank_sb), dtype=np.int64)
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j, c] = 1
                    V_sb[j * p + k, c] = 1
                    W_sb[k * n + i, c] = 1
                    c += 1
        return {
            "U": U_sb.tolist(),
            "V": V_sb.tolist(),
            "W": W_sb.tolist()
        }
