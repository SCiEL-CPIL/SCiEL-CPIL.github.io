import time
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 135

    def u_idx(i, j): return i * m + j
    def v_idx(i, j): return i * p + j
    def w_idx(i, j): return j * n + i

    U_cols = []
    V_cols = []
    W_cols = []

    def add_rank1_term(U_vec, V_vec, W_vec):
        U_cols.append(list(U_vec))
        V_cols.append(list(V_vec))
        W_cols.append(list(W_vec))

    def add_strassen(a_i, a_j, b_i, b_k, c_i, c_k):
        """
        Add 7 Strassen rank-1 terms for 2x2x2 block multiplication.
        a_i, a_j: start row/col of A block (2x2)
        b_i, b_k: start row/col of B block (2x2)
        c_i, c_k: start row/col of C block (2x2)
        """
        # M1: (a00+a11)(b00+b11) -> c00 + c11
        U1 = [0]*(n*m); V1 = [0]*(m*p); W1 = [0]*(p*n)
        U1[u_idx(a_i, a_j)] = 1; U1[u_idx(a_i+1, a_j+1)] = 1
        V1[v_idx(b_i, b_k)] = 1; V1[v_idx(b_i+1, b_k+1)] = 1
        W1[w_idx(c_i, c_k)] = 1; W1[w_idx(c_i+1, c_k+1)] = 1
        add_rank1_term(U1, V1, W1)

        # M2: (a10+a11)b00 -> c10 - c11
        U2 = [0]*(n*m); V2 = [0]*(m*p); W2 = [0]*(p*n)
        U2[u_idx(a_i+1, a_j)] = 1; U2[u_idx(a_i+1, a_j+1)] = 1
        V2[v_idx(b_i, b_k)] = 1
        W2[w_idx(c_i+1, c_k)] = 1; W2[w_idx(c_i+1, c_k+1)] = -1
        add_rank1_term(U2, V2, W2)

        # M3: a00(b01-b11) -> c01 + c11
        U3 = [0]*(n*m); V3 = [0]*(m*p); W3 = [0]*(p*n)
        U3[u_idx(a_i, a_j)] = 1
        V3[v_idx(b_i, b_k+1)] = 1; V3[v_idx(b_i+1, b_k+1)] = -1
        W3[w_idx(c_i, c_k+1)] = 1; W3[w_idx(c_i+1, c_k+1)] = 1
        add_rank1_term(U3, V3, W3)

        # M4: a11(b10-b00) -> c00 + c10
        U4 = [0]*(n*m); V4 = [0]*(m*p); W4 = [0]*(p*n)
        U4[u_idx(a_i+1, a_j+1)] = 1
        V4[v_idx(b_i+1, b_k)] = 1; V4[v_idx(b_i, b_k)] = -1
        W4[w_idx(c_i, c_k)] = 1; W4[w_idx(c_i+1, c_k)] = 1
        add_rank1_term(U4, V4, W4)

        # M5: (a00+a01)b11 -> -c00 + c01
        U5 = [0]*(n*m); V5 = [0]*(m*p); W5 = [0]*(p*n)
        U5[u_idx(a_i, a_j)] = 1; U5[u_idx(a_i, a_j+1)] = 1
        V5[v_idx(b_i+1, b_k+1)] = 1
        W5[w_idx(c_i, c_k)] = -1; W5[w_idx(c_i, c_k+1)] = 1
        add_rank1_term(U5, V5, W5)

        # M6: (a10-a00)(b00+b01) -> c11
        U6 = [0]*(n*m); V6 = [0]*(m*p); W6 = [0]*(p*n)
        U6[u_idx(a_i+1, a_j)] = 1; U6[u_idx(a_i, a_j)] = -1
        V6[v_idx(b_i, b_k)] = 1; V6[v_idx(b_i, b_k+1)] = 1
        W6[w_idx(c_i+1, c_k+1)] = 1
        add_rank1_term(U6, V6, W6)

        # M7: (a01-a11)(b10+b11) -> c00
        U7 = [0]*(n*m); V7 = [0]*(m*p); W7 = [0]*(p*n)
        U7[u_idx(a_i, a_j+1)] = 1; U7[u_idx(a_i+1, a_j+1)] = -1
        V7[v_idx(b_i+1, b_k)] = 1; V7[v_idx(b_i+1, b_k+1)] = 1
        W7[w_idx(c_i, c_k)] = 1
        add_rank1_term(U7, V7, W7)

    # 1. Term for j=2 (all i, all k)
    U_t = [0]*(n*m); V_t = [0]*(m*p); W_t = [0]*(p*n)
    for i in range(n):
        U_t[u_idx(i, 2)] = 1
    for k in range(p):
        V_t[v_idx(2, k)] = 1
    for i in range(n):
        for k in range(p):
            W_t[w_idx(i, k)] = 1
    add_rank1_term(U_t, V_t, W_t)

    # 2. Term for i=2, j=0,1 (all k)
    U_t = [0]*(n*m); V_t = [0]*(m*p); W_t = [0]*(p*n)
    U_t[u_idx(2, 0)] = 1
    U_t[u_idx(2, 1)] = 1
    for k in range(p):
        V_t[v_idx(0, k)] = 1
        V_t[v_idx(1, k)] = 1
        W_t[w_idx(2, k)] = 1
    add_rank1_term(U_t, V_t, W_t)

    # 3-9. Strassen for i=0,1; j=0,1; k=0,1
    add_strassen(0, 0, 0, 0, 0, 0)

    # 10-16. Strassen for i=0,1; j=0,1; k=2,3
    add_strassen(0, 0, 0, 2, 0, 2)

    # 17-23. Strassen for i=0,1; j=0,1; k=4,5
    add_strassen(0, 0, 0, 4, 0, 4)

    # 24. Rank 1 for i=0,1; j=0,1; k=6
    U_t = [0]*(n*m); V_t = [0]*(m*p); W_t = [0]*(p*n)
    for i in range(2):
        for j in range(2):
            U_t[u_idx(i, j)] = 1
    for j in range(2):
        V_t[v_idx(j, 6)] = 1
    for i in range(2):
        W_t[w_idx(i, 6)] = 1
    add_rank1_term(U_t, V_t, W_t)

    rank = len(U_cols)

    # Build matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for r in range(rank):
        for i in range(n * m):
            if U_cols[r][i] != 0:
                U[i][r] = U_cols[r][i]
        for i in range(m * p):
            if V_cols[r][i] != 0:
                V[i][r] = V_cols[r][i]
        for i in range(p * n):
            if W_cols[r][i] != 0:
                W[i][r] = W_cols[r][i]

    # Verification
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)

    def verify():
        np.random.seed(12345)
        for trial in range(5):
            A_test = np.random.randn(n, m)
            B_test = np.random.randn(m, p)
            C_true = A_test @ B_test

            C_comp = np.zeros((n, p))
            for r in range(rank):
                u_r = U_arr[:, r].reshape(n, m)
                v_r = V_arr[:, r].reshape(m, p)
                w_r = W_arr[:, r].reshape(p, n)
                C_comp += (A_test * u_r).sum(axis=1)[:, np.newaxis] * (B_test * v_r).sum(axis=0)[np.newaxis, :] * w_r.T

            if not np.allclose(C_comp, C_true, atol=1e-9):
                return False
        return True

    if not verify():
        # Fallback to pure schoolbook
        rank_fb = n * m * p
        U_fb = [[0] * rank_fb for _ in range(n * m)]
        V_fb = [[0] * rank_fb for _ in range(m * p)]
        W_fb = [[0] * rank_fb for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][col] = 1
                    V_fb[j * p + k][col] = 1
                    W_fb[k * n + i][col] = 1
                    col += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
