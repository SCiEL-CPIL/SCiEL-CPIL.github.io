"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0
    n, m, p = DIMENSIONS  # 4, 4, 8

    # Known best: use recursive Strassen (rank 98).
    # We will build it and verify; on failure, fall back to schoolbook.

    # Strassen 2x2x2 decomposition (indices for 2x2 matrices)
    # A indices: 0=A11, 1=A12, 2=A21, 3=A22
    # B indices: 0=B11, 1=B12, 2=B21, 3=B22
    # C indices: 0=C11, 1=C12, 2=C21, 3=C22

    strassen_2x2x2 = [
        ((1, 0, 0, 1), (1, 0, 0, 1), (1, -1, 1, -1)),
        ((0, 0, 1, 1), (1, 0, 0, 0), (0, 0, 1, 1)),
        ((1, 0, 0, 0), (0, 1, 0, -1), (0, 1, 0, -1)),
        ((0, 0, 0, 1), (0, 0, 1, -1), (-1, 0, 1, 0)),
        ((1, 1, 0, 0), (0, 0, 0, 1), (1, 1, 0, 0)),
        ((-1, 0, 1, 0), (1, 1, 0, 0), (0, 0, 1, 1)),
        ((0, 1, 0, -1), (0, 0, 1, 1), (0, 1, 0, 1)),
    ]

    # 4x4 block-level Strassen (coefficients for which 2x2 blocks combine)
    strassen_4x4_blocks = [
        ((1, 0, 0, 1), (1, 0, 0, 1), (1, 0, 0, 1)),
        ((0, 0, 1, 1), (1, 0, 0, 0), (0, 0, 1, 0)),
        ((1, 0, 0, 0), (0, 1, 0, -1), (0, 1, 0, 0)),
        ((0, 0, 0, 1), (0, 0, 1, -1), (0, 0, 1, 0)),
        ((1, 1, 0, 0), (0, 0, 0, 1), (1, 1, 0, 0)),
        ((-1, 0, 1, 0), (1, 1, 0, 0), (0, 0, 1, 0)),
        ((0, 1, 0, -1), (0, 0, 1, 1), (0, 1, 0, 0)),
    ]

    rank = 98
    U = np.zeros((16, rank))  # 4*4 = 16
    V = np.zeros((32, rank))  # 4*8 = 32
    W = np.zeros((32, rank))  # 8*4 = 32

    col = 0

    for half in range(2):
        for step in range(7):
            a_comb, b_comb, c_comb = strassen_4x4_blocks[step]
            for sub in range(7):
                u_sub, v_sub, w_sub = strassen_2x2x2[sub]

                # A: 4x4, blocks A11[0:2,0:2], A12[0:2,2:4], A21[2:4,0:2], A22[2:4,2:4]
                for bi in range(2):
                    for bj in range(2):
                        a_coeff = a_comb[bi * 2 + bj]
                        if a_coeff == 0:
                            continue
                        for li in range(2):
                            for lj in range(2):
                                local_coeff = u_sub[li * 2 + lj]
                                if local_coeff == 0:
                                    continue
                                global_i = bi * 2 + li
                                global_j = bj * 2 + lj
                                U[global_i * 4 + global_j, col] = a_coeff * local_coeff

                # B: 4x8, half*4 offset on columns
                for bi in range(2):
                    for bj in range(2):
                        b_coeff = b_comb[bi * 2 + bj]
                        if b_coeff == 0:
                            continue
                        for li in range(2):
                            for lj in range(2):
                                local_coeff = v_sub[li * 2 + lj]
                                if local_coeff == 0:
                                    continue
                                global_i = bi * 2 + li
                                global_j = half * 4 + bj * 2 + lj
                                V[global_i * 8 + global_j, col] = b_coeff * local_coeff

                # C: 8x4 (W index k*n+i = k*4+i), half*4 offset on columns
                for bi in range(2):
                    for bj in range(2):
                        c_coeff = c_comb[bi * 2 + bj]
                        if c_coeff == 0:
                            continue
                        for li in range(2):
                            for lj in range(2):
                                local_coeff = w_sub[li * 2 + lj]
                                if local_coeff == 0:
                                    continue
                                global_i = bi * 2 + li  # k
                                global_j = half * 4 + bj * 2 + lj  # i
                                W[global_i * 4 + global_j, col] = c_coeff * local_coeff

                col += 1

    # Verify with one random test
    A_test = np.random.randn(4, 4)
    B_test = np.random.randn(4, 8)
    C_test = A_test @ B_test

    # Vectorized check: C[i,k] = sum_c sum_j U[i,j,c] * V[j,k,c] * W[k,i,c]
    C_check = np.zeros((4, 8))
    for c in range(rank):
        u = U[:, c].reshape(4, 4)
        v = V[:, c].reshape(4, 8)
        w = W[:, c].reshape(8, 4)
        # For each i, k: C[i,k] += sum_j u[i,j]*v[j,k]*w[k,i]
        # = (u @ v) element-wise with w reshaped... 
        # Actually: for each k, sum_j u[i,j]*v[j,k] * w[k,i]
        # Let M = u @ v (4x8), then C[i,k] += M[i,k] * w[k,i]
        M = u @ v
        # w is 8x4, w[k,i] -> element (k,i)
        # M[i,k] * w[k,i] = M * w.T  (since w[k,i] = w.T[i,k])
        C_check += M * w.T

    if not np.allclose(C_test, C_check, atol=1e-10):
        # Fallback to schoolbook rank 128
        rank_fb = 128
        U_fb = np.zeros((16, rank_fb))
        V_fb = np.zeros((32, rank_fb))
        W_fb = np.zeros((32, rank_fb))
        col_fb = 0
        for i in range(4):
            for j in range(4):
                for k in range(8):
                    U_fb[i * 4 + j, col_fb] = 1
                    V_fb[j * 8 + k, col_fb] = 1
                    W_fb[k * 4 + i, col_fb] = 1
                    col_fb += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
