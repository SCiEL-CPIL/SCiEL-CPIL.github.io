import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin before 150s limit

    n, m, p = DIMENSIONS  # n=2, m=3, p=5

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
        U[(i_off+0)*m + (j_off+0), col] = 1.0
        U[(i_off+1)*m + (j_off+1), col] = 1.0
        V[(j_off+0)*p + (k_off+0), col] = 1.0
        V[(j_off+1)*p + (k_off+1), col] = 1.0
        W[(k_off+0)*n + (i_off+0), col] = 1.0
        W[(k_off+1)*n + (i_off+1), col] = 1.0
        col += 1

        U[(i_off+1)*m + (j_off+0), col] = 1.0
        U[(i_off+1)*m + (j_off+1), col] = 1.0
        V[(j_off+0)*p + (k_off+0), col] = 1.0
        W[(k_off+0)*n + (i_off+1), col] = 1.0
        W[(k_off+1)*n + (i_off+1), col] = -1.0
        col += 1

        U[(i_off+0)*m + (j_off+0), col] = 1.0
        V[(j_off+0)*p + (k_off+1), col] = 1.0
        V[(j_off+1)*p + (k_off+1), col] = -1.0
        W[(k_off+1)*n + (i_off+0), col] = 1.0
        W[(k_off+1)*n + (i_off+1), col] = 1.0
        col += 1

        U[(i_off+1)*m + (j_off+1), col] = 1.0
        V[(j_off+1)*p + (k_off+0), col] = 1.0
        V[(j_off+0)*p + (k_off+0), col] = -1.0
        W[(k_off+0)*n + (i_off+0), col] = 1.0
        W[(k_off+0)*n + (i_off+1), col] = 1.0
        col += 1

        U[(i_off+0)*m + (j_off+0), col] = 1.0
        U[(i_off+0)*m + (j_off+1), col] = 1.0
        V[(j_off+1)*p + (k_off+1), col] = 1.0
        W[(k_off+0)*n + (i_off+0), col] = -1.0
        W[(k_off+1)*n + (i_off+0), col] = 1.0
        col += 1

        U[(i_off+1)*m + (j_off+0), col] = 1.0
        U[(i_off+0)*m + (j_off+0), col] = -1.0
        V[(j_off+0)*p + (k_off+0), col] = 1.0
        V[(j_off+0)*p + (k_off+1), col] = 1.0
        W[(k_off+1)*n + (i_off+1), col] = 1.0
        col += 1

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

    # Search for rank 27 by checking linear dependence
    M = np.zeros((n*m*p, rank))
    for t in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i * (m * p) + j * p + k
                    M[idx, t] = U[i * m + j, t] * V[j * p + k, t] * W[k * n + i, t]
    
    try:
        U_svd, s, Vh = np.linalg.svd(M, full_matrices=False)
        tol = 1e-10
        rank_M = np.sum(s > tol)
        
        if rank_M < rank:
            null_space = Vh[rank_M:, :]
            for i in range(rank):
                for vec in null_space:
                    if abs(vec[i]) > 1e-10:
                        c = -vec / vec[i]
                        if np.allclose(M @ c, 0, atol=1e-9):
                            new_rank = rank - 1
                            U_new = np.zeros((n * m, new_rank), dtype=float)
                            V_new = np.zeros((m * p, new_rank), dtype=float)
                            W_new = np.zeros((p * n, new_rank), dtype=float)
                            col_new = 0
                            for j in range(rank):
                                if j == i:
                                    continue
                                coeff = 1.0 + c[j]
                                if abs(coeff) < 1e-10:
                                    continue
                                U_new[:, col_new] = coeff * U[:, j]
                                V_new[:, col_new] = V[:, j]
                                W_new[:, col_new] = W[:, j]
                                col_new += 1
                            
                            if col_new < new_rank:
                                U_new = U_new[:, :col_new]
                                V_new = V_new[:, :col_new]
                                W_new = W_new[:, :col_new]
                                new_rank = col_new
                                
                            np.random.seed(42)
                            A_test = np.random.randn(n, m)
                            B_test = np.random.randn(m, p)
                            C_test = A_test @ B_test
                            C_decomp = np.zeros((n, p))
                            for t in range(new_rank):
                                u_t = U_new[:, t].reshape(n, m)
                                v_t = V_new[:, t].reshape(m, p)
                                w_t = W_new[:, t].reshape(p, n)
                                s1 = np.sum(A_test * u_t)
                                s2 = np.sum(B_test * v_t)
                                C_decomp += s1 * s2 * w_t.T

                            if np.allclose(C_test, C_decomp, atol=1e-10):
                                return {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}
                            else:
                                break
                        break
                break
    except Exception:
        pass

    # Verify original decomposition
    np.random.seed(42)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_test = A_test @ B_test
    C_decomp = np.zeros((n, p))
    for t in range(rank):
        u_t = U[:, t].reshape(n, m)
        v_t = V[:, t].reshape(m, p)
        w_t = W[:, t].reshape(p, n)
        s1 = np.sum(A_test * u_t)
        s2 = np.sum(B_test * v_t)
        C_decomp += s1 * s2 * w_t.T

    if not np.allclose(C_test, C_decomp, atol=1e-10):
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
        return {"U": U_sb, "V": V_sb, "W": W_sb}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
