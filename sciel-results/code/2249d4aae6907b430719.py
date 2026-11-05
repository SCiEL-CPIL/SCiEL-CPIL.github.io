import time
import numpy as np

START_TIME = time.time()
TIME_LIMIT = 110  # Safety margin for 120s limit

def construct():
    n, m, p = 3, 5, 7

    # Block-partition strategy:
    #  - columns 0,1 and 2,3 (the two middle groups of 2) handled with Strassen
    #    2x2x2 blocks (rank 7 each) plus a 2x2x1 tail (rank 4)
    #  - column 4 (single middle) handled with a 3x1x7 schoolbook block (rank 21)
    # Total rank = 39 + 39 + 21 = 99
    rank_total = 99

    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]

    def add_strassen(rank_offset, i0, j0, k0):
        # M222 Strassen for A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2]
        u_loc = [[0] * 7 for _ in range(4)]
        v_loc = [[0] * 7 for _ in range(4)]
        w_loc = [[0] * 7 for _ in range(4)]

        # M1 = (A11+A22)*(B11+B22)
        u_loc[0][0] = 1; u_loc[3][0] = 1
        v_loc[0][0] = 1; v_loc[3][0] = 1
        w_loc[0][0] = 1; w_loc[3][0] = 1

        # M2 = (A21+A22)*B11
        u_loc[2][1] = 1; u_loc[3][1] = 1
        v_loc[0][1] = 1
        w_loc[1][1] = 1; w_loc[3][1] = -1

        # M3 = A11*(B21-B22)
        u_loc[0][2] = 1
        v_loc[1][2] = 1; v_loc[3][2] = -1
        w_loc[2][2] = 1; w_loc[3][2] = 1

        # M4 = A22*(B11-B21)
        u_loc[3][3] = 1
        v_loc[2][3] = 1; v_loc[0][3] = -1
        w_loc[0][3] = 1; w_loc[1][3] = 1

        # M5 = (A11+A12)*B22
        u_loc[0][4] = 1; u_loc[1][4] = 1
        v_loc[3][4] = 1
        w_loc[0][4] = -1; w_loc[2][4] = 1

        # M6 = (A21-A11)*(B11+B12)
        u_loc[2][5] = 1; u_loc[0][5] = -1
        v_loc[0][5] = 1; v_loc[1][5] = 1
        w_loc[3][5] = 1

        # M7 = (A12-A22)*(B21+B22)
        u_loc[1][6] = 1; u_loc[3][6] = -1
        v_loc[2][6] = 1; v_loc[3][6] = 1
        w_loc[0][6] = 1

        for i_l in range(2):
            for j_l in range(2):
                u_g_idx = (i0 + i_l) * m + (j0 + j_l)
                for r in range(7):
                    U[u_g_idx][rank_offset + r] += u_loc[i_l * 2 + j_l][r]
        for j_l in range(2):
            for k_l in range(2):
                v_g_idx = (j0 + j_l) * p + (k0 + k_l)
                for r in range(7):
                    V[v_g_idx][rank_offset + r] += v_loc[j_l * 2 + k_l][r]
        for k_l in range(2):
            for i_l in range(2):
                w_g_idx = (k0 + k_l) * n + (i0 + i_l)
                for r in range(7):
                    W[w_g_idx][rank_offset + r] += w_loc[k_l * 2 + i_l][r]

    def add_m221(rank_offset, i0, j0, k0):
        # M221 for A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+1]  (schoolbook, rank 4)
        u_loc = [[0] * 4 for _ in range(4)]
        v_loc = [[0] * 4 for _ in range(2)]
        w_loc = [[0] * 4 for _ in range(2)]

        u_loc[0][0] = 1; v_loc[0][0] = 1; w_loc[0][0] = 1   # A00 B00
        u_loc[1][1] = 1; v_loc[1][1] = 1; w_loc[0][1] = 1   # A01 B10
        u_loc[2][2] = 1; v_loc[0][2] = 1; w_loc[1][2] = 1   # A10 B00
        u_loc[3][3] = 1; v_loc[1][3] = 1; w_loc[1][3] = 1   # A11 B10

        for i_l in range(2):
            for j_l in range(2):
                u_g_idx = (i0 + i_l) * m + (j0 + j_l)
                for r in range(4):
                    U[u_g_idx][rank_offset + r] += u_loc[i_l * 2 + j_l][r]
        for j_l in range(2):
            for k_l in range(1):
                v_g_idx = (j0 + j_l) * p + (k0 + k_l)
                for r in range(4):
                    V[v_g_idx][rank_offset + r] += v_loc[j_l * 1 + k_l][r]
        for k_l in range(1):
            for i_l in range(2):
                w_g_idx = (k0 + k_l) * n + (i0 + i_l)
                for r in range(4):
                    W[w_g_idx][rank_offset + r] += w_loc[k_l * 2 + i_l][r]

    def add_m127(rank_offset, i0, j0, k0):
        # M127 for A[i0:i0+1, j0:j0+2] * B[j0:j0+2, k0:k0+7]  (schoolbook, rank 14)
        for j_l in range(2):
            for k_l in range(7):
                r = j_l * 7 + k_l
                u_g_idx = (i0 + 0) * m + (j0 + j_l)
                U[u_g_idx][rank_offset + r] += 1
                v_g_idx = (j0 + j_l) * p + (k0 + k_l)
                V[v_g_idx][rank_offset + r] += 1
                w_g_idx = (k0 + k_l) * n + (i0 + 0)
                W[w_g_idx][rank_offset + r] += 1

    def add_m317(rank_offset, i0, j0, k0):
        # M317 for A[i0:i0+3, j0:j0+1] * B[j0:j0+1, k0:k0+7]  (schoolbook, rank 21)
        for i_l in range(3):
            for k_l in range(7):
                r = i_l * 7 + k_l
                u_g_idx = (i0 + i_l) * m + (j0 + 0)
                U[u_g_idx][rank_offset + r] += 1
                v_g_idx = (j0 + 0) * p + (k0 + k_l)
                V[v_g_idx][rank_offset + r] += 1
                w_g_idx = (k0 + k_l) * n + (i0 + i_l)
                W[w_g_idx][rank_offset + r] += 1

    # ---- assemble blocks ----
    # Block 1: middle cols 0,1  (rank 39)
    add_strassen(0, 0, 0, 0)
    add_strassen(7, 0, 0, 2)
    add_strassen(14, 0, 0, 4)
    add_m221(21, 0, 0, 6)
    add_m127(25, 2, 0, 0)

    # Block 2: middle cols 2,3  (rank 39)
    add_strassen(39, 0, 2, 0)
    add_strassen(46, 0, 2, 2)
    add_strassen(53, 0, 2, 4)
    add_m221(60, 0, 2, 6)
    add_m127(64, 2, 2, 0)

    # Block 3: middle col 4  (rank 21)
    add_m317(78, 0, 4, 0)

    # ---- verification / polish pass ----
    # Verify the decomposition reproduces the matrix-multiplication tensor,
    # repair any residual errors, and drop exactly-zero rank columns.
    try:
        if time.time() - START_TIME < TIME_LIMIT:
            U_np = np.array(U, dtype=np.float64)
            V_np = np.array(V, dtype=np.float64)
            W_np = np.array(W, dtype=np.float64)

            # residual[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]  (should be 1 for all)
            resid = np.zeros((n, m, p), dtype=np.float64)
            for r in range(rank_total):
                u = U_np[:, r].reshape(n, m)
                v = V_np[:, r].reshape(m, p)
                w = W_np[:, r].reshape(p, n)
                resid += np.einsum('ij,jk,ki->ijk', u, v, w)

            bad = np.where(np.abs(resid - 1.0) > 1e-9)
            if len(bad[0]) > 0:
                # Fallback: guarantee correctness with a full schoolbook decomposition
                rank_total = n * m * p
                U = [[0] * rank_total for _ in range(n * m)]
                V = [[0] * rank_total for _ in range(m * p)]
                W = [[0] * rank_total for _ in range(p * n)]
                idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            U[i * m + j][idx] = 1
                            V[j * p + k][idx] = 1
                            W[k * n + i][idx] = 1
                            idx += 1
                return {"U": U, "V": V, "W": W}

            # Correct: drop exactly-zero rank columns to shave rank
            U_np = np.array(U, dtype=np.float64)
            V_np = np.array(V, dtype=np.float64)
            W_np = np.array(W, dtype=np.float64)
            col_norm = (np.sum(np.abs(U_np), axis=0) +
                        np.sum(np.abs(V_np), axis=0) +
                        np.sum(np.abs(W_np), axis=0))
            keep = [r for r in range(rank_total) if col_norm[r] > 1e-12]
            if len(keep) < rank_total:
                U = [[U_np[i, r] for r in keep] for i in range(n * m)]
                V = [[V_np[i, r] for r in keep] for i in range(m * p)]
                W = [[W_np[i, r] for r in keep] for i in range(p * n)]
    except Exception:
        pass

    return {"U": U, "V": V, "W": W}
