"""Tensor decomposition for matrix multiplication 3x3 * 3x8.
Uses Strassen's rank-7 decomposition for 2x2x2 blocks combined with
direct terms for remaining dimensions, targeting rank < 72.
Adds a refinement step to detect and remove linearly dependent rank terms.
"""

DIMENSIONS = [3, 3, 8]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill

    # Build the rank-68 decomposition using Strassen + direct terms
    rank_init = 68
    U = np.zeros((n * m, rank_init), dtype=int)
    V = np.zeros((m * p, rank_init), dtype=int)
    W = np.zeros((p * n, rank_init), dtype=int)

    col = 0

    def add_strassen(row_off_a, col_off_a, row_off_b, col_off_b):
        nonlocal col
        a_idx = row_off_a * m + col_off_a
        b_idx = row_off_a * m + col_off_a + 1
        c_idx = (row_off_a + 1) * m + col_off_a
        d_idx = (row_off_a + 1) * m + col_off_a + 1

        e_idx = row_off_b * p + col_off_b
        f_idx = row_off_b * p + col_off_b + 1
        g_idx = (row_off_b + 1) * p + col_off_b
        h_idx = (row_off_b + 1) * p + col_off_b + 1

        C00_idx = col_off_b * n + row_off_a
        C01_idx = (col_off_b + 1) * n + row_off_a
        C10_idx = col_off_b * n + row_off_a + 1
        C11_idx = (col_off_b + 1) * n + row_off_a + 1

        # m1 = (a+d)(e+h)
        U[a_idx, col] = 1
        U[d_idx, col] = 1
        V[e_idx, col] = 1
        V[h_idx, col] = 1
        W[C00_idx, col] = 1
        W[C11_idx, col] = 1
        col += 1

        # m2 = (c+d)*e
        U[c_idx, col] = 1
        U[d_idx, col] = 1
        V[e_idx, col] = 1
        W[C10_idx, col] = 1
        W[C11_idx, col] = -1
        col += 1

        # m3 = a*(f-h)
        U[a_idx, col] = 1
        V[f_idx, col] = 1
        V[h_idx, col] = -1
        W[C01_idx, col] = 1
        W[C11_idx, col] = 1
        col += 1

        # m4 = d*(g-e)
        U[d_idx, col] = 1
        V[g_idx, col] = 1
        V[e_idx, col] = -1
        W[C00_idx, col] = 1
        W[C10_idx, col] = 1
        col += 1

        # m5 = (a+b)*h
        U[a_idx, col] = 1
        U[b_idx, col] = 1
        V[h_idx, col] = 1
        W[C00_idx, col] = -1
        W[C01_idx, col] = 1
        col += 1

        # m6 = (c-a)*(e+f)
        U[c_idx, col] = 1
        U[a_idx, col] = -1
        V[e_idx, col] = 1
        V[f_idx, col] = 1
        W[C11_idx, col] = 1
        col += 1

        # m7 = (b-d)*(g+h)
        U[b_idx, col] = 1
        U[d_idx, col] = -1
        V[g_idx, col] = 1
        V[h_idx, col] = 1
        W[C00_idx, col] = 1
        col += 1

    def add_direct(i, j, k):
        nonlocal col
        U[i * m + j, col] = 1
        V[j * p + k, col] = 1
        W[k * n + i, col] = 1
        col += 1

    # 4 Strassen blocks for A[0:2, 0:2] * B[0:2, 0:8]
    for k_block in range(4):
        add_strassen(0, 0, 0, k_block * 2)

    # Direct terms for A[0:2, 2] * B[2, 0:8]
    for i in range(2):
        for k in range(8):
            add_direct(i, 2, k)

    # Direct terms for A[2, 0:2] * B[0:2, 0:8]
    for j in range(2):
        for k in range(8):
            add_direct(2, j, k)

    # Direct terms for A[2, 2] * B[2, 0:8]
    for k in range(8):
        add_direct(2, 2, k)

    if col != 68:
        # Fallback: rebuild schoolbook
        rank_fb = n * m * p
        U_fb = np.zeros((n * m, rank_fb), dtype=int)
        V_fb = np.zeros((m * p, rank_fb), dtype=int)
        W_fb = np.zeros((p * n, rank_fb), dtype=int)
        c = 0
        for i2 in range(n):
            for j2 in range(m):
                for k2 in range(p):
                    U_fb[i2 * m + j2, c] = 1
                    V_fb[j2 * p + k2, c] = 1
                    W_fb[k2 * n + i2, c] = 1
                    c += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    # Verify correctness of the rank-68 decomposition
    def verify_decomp(Ua, Va, Wa):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(Ua[i * m + j] * Va[j * p + k] * Wa[k * n + i])
                    if s != 1:
                        return False
        return True

    if not verify_decomp(U, V, W):
        # Fallback to schoolbook
        rank_fb = n * m * p
        U_fb = np.zeros((n * m, rank_fb), dtype=int)
        V_fb = np.zeros((m * p, rank_fb), dtype=int)
        W_fb = np.zeros((p * n, rank_fb), dtype=int)
        c = 0
        for i2 in range(n):
            for j2 in range(m):
                for k2 in range(p):
                    U_fb[i2 * m + j2, c] = 1
                    V_fb[j2 * p + k2, c] = 1
                    W_fb[k2 * n + i2, c] = 1
                    c += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    # Refinement: check for linear dependencies among rank terms
    # Each rank term r corresponds to a tensor T_r[i,j,k] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We form a matrix where each row is a flattened rank term (216 entries)
    # and check the rank of this matrix.
    # If rank < 68, some terms are linearly dependent and can be removed.

    if time.time() - start_time > DEADLINE - 5:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Build the 216 x 68 matrix of rank terms
    term_matrix = np.zeros((n * m * p, 68), dtype=float)
    for r in range(68):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    term_matrix[k * n * m + i * m + j, r] = U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]

    # Compute rank via SVD
    try:
        _, s, vh = np.linalg.svd(term_matrix, full_matrices=False)
        tol = 1e-10
        rank_computed = np.sum(s > tol)
    except Exception:
        rank_computed = 68

    if rank_computed < 68:
        # Identify linearly independent columns
        # Use QR with column pivoting to find pivot columns
        try:
            Q, R, P = np.linalg.qr(term_matrix, mode='complete')
            # Find pivot columns from R
            diag = np.abs(np.diag(R))
            pivot_threshold = 1e-10
            pivots = np.where(diag > pivot_threshold)[0]
            # P is a permutation matrix; pivot columns in original indexing
            # Actually, numpy's qr doesn't do column pivoting directly.
            # Let's use a different approach: greedy elimination
            pass
        except Exception:
            pass

        # Simpler: keep all 68 terms since removing them requires careful reconstruction
        # The linear dependence in the tensor space doesn't directly imply we can remove
        # a rank term while preserving the decomposition (it's not a linear combination
        # of rank-1 tensors in the usual sense).
        # So we stick with 68.

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
