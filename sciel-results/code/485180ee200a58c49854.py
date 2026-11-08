import time
import numpy as np

DIMENSIONS = [3, 3, 7]

def construct():
    start_time = time.time()
    deadline = 145.0
    safety_margin = 5.0

    n, m, p = DIMENSIONS

    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(i, k): return k * n + i

    terms = []

    def add_term(u_vec, v_vec, w_vec):
        terms.append((u_vec, v_vec, w_vec))

    def zero_U(): return [0] * (n * m)
    def zero_V(): return [0] * (m * p)
    def zero_W(): return [0] * (p * n)

    def strassen_2x2(a_i, a_j, b_j, b_k, c_i, c_k):
        U1, V1, W1 = zero_U(), zero_V(), zero_W()
        U1[u_idx(a_i, a_j)] = 1; U1[u_idx(a_i+1, a_j+1)] = 1
        V1[v_idx(b_j, b_k)] = 1; V1[v_idx(b_j+1, b_k+1)] = 1
        W1[w_idx(c_i, c_k)] = 1; W1[w_idx(c_i+1, c_k+1)] = 1
        add_term(U1, V1, W1)

        U2, V2, W2 = zero_U(), zero_V(), zero_W()
        U2[u_idx(a_i+1, a_j)] = 1; U2[u_idx(a_i+1, a_j+1)] = 1
        V2[v_idx(b_j, b_k)] = 1
        W2[w_idx(c_i+1, c_k)] = 1; W2[w_idx(c_i+1, c_k+1)] = -1
        add_term(U2, V2, W2)

        U3, V3, W3 = zero_U(), zero_V(), zero_W()
        U3[u_idx(a_i, a_j)] = 1
        V3[v_idx(b_j, b_k+1)] = 1; V3[v_idx(b_j+1, b_k+1)] = -1
        W3[w_idx(c_i, c_k+1)] = 1; W3[w_idx(c_i+1, c_k+1)] = 1
        add_term(U3, V3, W3)

        U4, V4, W4 = zero_U(), zero_V(), zero_W()
        U4[u_idx(a_i+1, a_j+1)] = 1
        V4[v_idx(b_j+1, b_k)] = 1; V4[v_idx(b_j, b_k)] = -1
        W4[w_idx(c_i, c_k)] = 1; W4[w_idx(c_i+1, c_k)] = 1
        add_term(U4, V4, W4)

        U5, V5, W5 = zero_U(), zero_V(), zero_W()
        U5[u_idx(a_i, a_j)] = 1; U5[u_idx(a_i, a_j+1)] = 1
        V5[v_idx(b_j+1, b_k+1)] = 1
        W5[w_idx(c_i, c_k)] = -1; W5[w_idx(c_i, c_k+1)] = 1
        add_term(U5, V5, W5)

        U6, V6, W6 = zero_U(), zero_V(), zero_W()
        U6[u_idx(a_i+1, a_j)] = 1; U6[u_idx(a_i, a_j)] = -1
        V6[v_idx(b_j, b_k)] = 1; V6[v_idx(b_j, b_k+1)] = 1
        W6[w_idx(c_i+1, c_k+1)] = 1
        add_term(U6, V6, W6)

        U7, V7, W7 = zero_U(), zero_V(), zero_W()
        U7[u_idx(a_i, a_j+1)] = 1; U7[u_idx(a_i+1, a_j+1)] = -1
        V7[v_idx(b_j+1, b_k)] = 1; V7[v_idx(b_j+1, b_k+1)] = 1
        W7[w_idx(c_i, c_k)] = 1
        add_term(U7, V7, W7)

    def schoolbook_block(a_i, a_j, b_j, b_k, c_i, c_k, di, dj, dk):
        for ii in range(di):
            for jj in range(dj):
                for kk in range(dk):
                    U, V, W = zero_U(), zero_V(), zero_W()
                    U[u_idx(a_i+ii, a_j+jj)] = 1
                    V[v_idx(b_j+jj, b_k+kk)] = 1
                    W[w_idx(c_i+ii, c_k+kk)] = 1
                    add_term(U, V, W)

    for k_start in [0, 2, 4]:
        strassen_2x2(0, 0, 0, k_start, 0, k_start)
        schoolbook_block(0, 2, 2, k_start, 0, k_start, 2, 1, 2)
        schoolbook_block(2, 0, 0, k_start, 2, k_start, 1, 2, 2)
        schoolbook_block(2, 2, 2, k_start, 2, k_start, 1, 1, 2)

    schoolbook_block(0, 0, 0, 6, 0, 6, 2, 2, 1)
    schoolbook_block(0, 2, 2, 6, 0, 6, 2, 1, 1)
    schoolbook_block(2, 0, 0, 6, 2, 6, 1, 2, 1)
    schoolbook_block(2, 2, 2, 6, 2, 6, 1, 1, 1)

    rank = len(terms)
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    for c, (u_vec, v_vec, w_vec) in enumerate(terms):
        for i, val in enumerate(u_vec):
            if val != 0:
                U[i, c] = val
        for i, val in enumerate(v_vec):
            if val != 0:
                V[i, c] = val
        for i, val in enumerate(w_vec):
            if val != 0:
                W[i, c] = val

    U = U.tolist()
    V = V.tolist()
    W = W.tolist()

    if time.time() < deadline - safety_margin:
        def verify():
            U_arr = np.array(U, dtype=float)
            V_arr = np.array(V, dtype=float)
            W_arr = np.array(W, dtype=float)
            A_test = np.random.randn(n, m)
            B_test = np.random.randn(m, p)
            C_true = A_test @ B_test
            C_comp = np.zeros((n, p))
            for r in range(rank):
                u_mat = U_arr[:, r].reshape(n, m)
                v_mat = V_arr[:, r].reshape(m, p)
                w_mat = W_arr[:, r].reshape(p, n).T
                C_comp += u_mat @ v_mat * w_mat
            return np.allclose(C_comp, C_true, atol=1e-9)

        if not verify():
            rank = n * m * p
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i*m+j][col] = 1
                        V[j*p+k][col] = 1
                        W[k*n+i][col] = 1
                        col += 1

    return {"U": U, "V": V, "W": W}
