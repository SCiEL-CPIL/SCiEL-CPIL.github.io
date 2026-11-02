import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=5

    def verify(U_mat, V_mat, W_mat, r):
        for seed in [42, 123, 999]:
            np.random.seed(seed)
            A_test = np.random.randn(n, m)
            B_test = np.random.randn(m, p)
            C_test = A_test @ B_test
            C_decomp = np.zeros((n, p))
            for t in range(r):
                u_t = U_mat[:, t].reshape(n, m)
                v_t = V_mat[:, t].reshape(m, p)
                w_t = W_mat[:, t].reshape(p, n)
                C_decomp += np.sum(A_test * u_t) * np.sum(B_test * v_t) * w_t.T
            if not np.allclose(C_test, C_decomp, atol=1e-9):
                return False
        return True

    def build_rank_28():
        rank = 28
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0

        # Block 1: j=0, k=0..4 (schoolbook 2x1x5)
        for i in range(n):
            for k in range(p):
                U[i * m + 0, col] = 1.0
                V[0 * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1

        def add_strassen(i_off, j_off, k_off):
            nonlocal col
            # 1
            U[(i_off+0)*m + (j_off+0), col] = 1.0
            U[(i_off+1)*m + (j_off+1), col] = 1.0
            V[(j_off+0)*p + (k_off+0), col] = 1.0
            V[(j_off+1)*p + (k_off+1), col] = 1.0
            W[(k_off+0)*n + (i_off+0), col] = 1.0
            W[(k_off+1)*n + (i_off+1), col] = 1.0
            col += 1
            # 2
            U[(i_off+1)*m + (j_off+0), col] = 1.0
            U[(i_off+1)*m + (j_off+1), col] = 1.0
            V[(j_off+0)*p + (k_off+0), col] = 1.0
            W[(k_off+0)*n + (i_off+1), col] = 1.0
            W[(k_off+1)*n + (i_off+1), col] = -1.0
            col += 1
            # 3
            U[(i_off+0)*m + (j_off+0), col] = 1.0
            V[(j_off+0)*p + (k_off+1), col] = 1.0
            V[(j_off+1)*p + (k_off+1), col] = -1.0
            W[(k_off+1)*n + (i_off+0), col] = 1.0
            W[(k_off+1)*n + (i_off+1), col] = 1.0
            col += 1
            # 4
            U[(i_off+1)*m + (j_off+1), col] = 1.0
            V[(j_off+1)*p + (k_off+0), col] = 1.0
            V[(j_off+0)*p + (k_off+0), col] = -1.0
            W[(k_off+0)*n + (i_off+0), col] = 1.0
            W[(k_off+0)*n + (i_off+1), col] = 1.0
            col += 1
            # 5
            U[(i_off+0)*m + (j_off+0), col] = 1.0
            U[(i_off+0)*m + (j_off+1), col] = 1.0
            V[(j_off+1)*p + (k_off+1), col] = 1.0
            W[(k_off+0)*n + (i_off+0), col] = -1.0
            W[(k_off+1)*n + (i_off+0), col] = 1.0
            col += 1
            # 6
            U[(i_off+1)*m + (j_off+0), col] = 1.0
            U[(i_off+0)*m + (j_off+0), col] = -1.0
            V[(j_off+0)*p + (k_off+0), col] = 1.0
            V[(j_off+0)*p + (k_off+1), col] = 1.0
            W[(k_off+1)*n + (i_off+1), col] = 1.0
            col += 1
            # 7
            U[(i_off+0)*m + (j_off+1), col] = 1.0
            U[(i_off+1)*m + (j_off+1), col] = -1.0
            V[(j_off+1)*p + (k_off+0), col] = 1.0
            V[(j_off+1)*p + (k_off+1), col] = 1.0
            W[(k_off+0)*n + (i_off+0), col] = 1.0
            col += 1

        add_strassen(i_off=0, j_off=1, k_off=0)

        for i in range(n):
            for j in [1, 2]:
                U[i * m + j, col] = 1.0
                V[j * p + 2, col] = 1.0
                W[2 * n + i, col] = 1.0
                col += 1

        add_strassen(i_off=0, j_off=1, k_off=3)

        assert col == rank
        return U, V, W

    U28, V28, W28 = build_rank_28()
    
    if verify(U28, V28, W28, 28):
        return {"U": U28.tolist(), "V": V28.tolist(), "W": W28.tolist()}
    
    # Fallback: schoolbook (should not happen)
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb), dtype=float)
    V_sb = np.zeros((m * p, rank_sb), dtype=float)
    W_sb = np.zeros((p * n, rank_sb), dtype=float)
    col_sb = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col_sb] = 1.0
                V_sb[j * p + k, col_sb] = 1.0
                W_sb[k * n + i, col_sb] = 1.0
                col_sb += 1
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
