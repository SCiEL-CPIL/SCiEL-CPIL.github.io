"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 5
    deadline = time.time() + 115.0

    def schoolbook():
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

    def strassen_2x2x2():
        U = [
            [1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0],
            [0, 0, 1, 0, 0, 1, 0],
            [1, 1, 0, 0, 0, 0, 1],
        ]
        V = [
            [1, 0, 0, 0, 0, 1, 1],
            [0, 1, -1, 1, 0, 0, 1],
            [1, 0, 1, 0, 0, 1, 0],
            [1, 1, 0, -1, 1, 0, 0],
        ]
        W = [
            [1, 0, 1, 0, 0, 0, 0],
            [0, 1, 0, 0, 1, 0, 1],
            [0, 0, 0, 1, 0, 1, 0],
            [1, 0, 0, 1, -1, 0, 0],
        ]
        return U, V, W

    def verify_fast(U, V, W, n, m, p, tol=1e-9):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        U_r = U.reshape(n, m, rank)
        V_r = V.reshape(m, p, rank)
        W_r = W.reshape(p, n, rank)
        T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r, optimize=True)
        return np.allclose(T, np.ones((n, m, p)), atol=tol, rtol=0)

    best_result = schoolbook()
    best_rank = n * m * p

    U_cols = []
    V_cols = []
    W_cols = []

    # Block 1: i in {0,1}, j in {0,1} -> 2x2x5
    # Split k into {0,1}, {2,3}, {4}
    # k in {0,1}: 2x2x2 Strassen
    U_s, V_s, W_s = strassen_2x2x2()
    for c in range(7):
        U_col = [0.0] * (n * m)
        V_col = [0.0] * (m * p)
        W_col = [0.0] * (p * n)
        for i in range(2):
            for j in range(2):
                U_col[i*m + j] = U_s[i*2+j][c]
        for j in range(2):
            for k in range(2):
                V_col[j*p + k] = V_s[j*2+k][c]
        for k in range(2):
            for i in range(2):
                W_col[k*n + i] = W_s[k*2+i][c]
        U_cols.append(U_col)
        V_cols.append(V_col)
        W_cols.append(W_col)

    # k in {2,3}: 2x2x2 Strassen
    U_s, V_s, W_s = strassen_2x2x2()
    for c in range(7):
        U_col = [0.0] * (n * m)
        V_col = [0.0] * (m * p)
        W_col = [0.0] * (p * n)
        for i in range(2):
            for j in range(2):
                U_col[i*m + j] = U_s[i*2+j][c]
        for j in range(2):
            for k in range(2):
                V_col[j*p + (2+k)] = V_s[j*2+k][c]
        for k in range(2):
            for i in range(2):
                W_col[(2+k)*n + i] = W_s[k*2+i][c]
        U_cols.append(U_col)
        V_cols.append(V_col)
        W_cols.append(W_col)

    # k in {4}: 2x2x1 Rank 1
    U_col = [0.0] * (n * m)
    V_col = [0.0] * (m * p)
    W_col = [0.0] * (p * n)
    for i in range(2):
        for j in range(2):
            U_col[i*m + j] = 1.0
    for j in range(2):
        V_col[j*p + 4] = 1.0
    for i in range(2):
        W_col[4*n + i] = 1.0
    U_cols.append(U_col)
    V_cols.append(V_col)
    W_cols.append(W_col)

    # Block 2: i in {0,1}, j in {2,3} -> 2x2x5
    # k in {0,1}: 2x2x2 Strassen
    U_s, V_s, W_s = strassen_2x2x2()
    for c in range(7):
        U_col = [0.0] * (n * m)
        V_col = [0.0] * (m * p)
        W_col = [0.0] * (p * n)
        for i in range(2):
            for j in range(2):
                U_col[i*m + (2+j)] = U_s[i*2+j][c]
        for j in range(2):
            for k in range(2):
                V_col[(2+j)*p + k] = V_s[j*2+k][c]
        for k in range(2):
            for i in range(2):
                W_col[k*n + i] = W_s[k*2+i][c]
        U_cols.append(U_col)
        V_cols.append(V_col)
        W_cols.append(W_col)

    # k in {2,3}: 2x2x2 Strassen
    U_s, V_s, W_s = strassen_2x2x2()
    for c in range(7):
        U_col = [0.0] * (n * m)
        V_col = [0.0] * (m * p)
        W_col = [0.0] * (p * n)
        for i in range(2):
            for j in range(2):
                U_col[i*m + (2+j)] = U_s[i*2+j][c]
        for j in range(2):
            for k in range(2):
                V_col[(2+j)*p + (2+k)] = V_s[j*2+k][c]
        for k in range(2):
            for i in range(2):
                W_col[(2+k)*n + i] = W_s[k*2+i][c]
        U_cols.append(U_col)
        V_cols.append(V_col)
        W_cols.append(W_col)

    # k in {4}: 2x2x1 Rank 1
    U_col = [0.0] * (n * m)
    V_col = [0.0] * (m * p)
    W_col = [0.0] * (p * n)
    for i in range(2):
        for j in range(2):
            U_col[i*m + (2+j)] = 1.0
    for j in range(2):
        V_col[(2+j)*p + 4] = 1.0
    for i in range(2):
        W_col[4*n + i] = 1.0
    U_cols.append(U_col)
    V_cols.append(V_col)
    W_cols.append(W_col)

    # Block 3: i=2, j in {0,3} -> 1x2x5
    # Use rank-1 decomposition for each k, but combine where possible
    # For 1x2x5, we can use 5 rank-1 terms (one per k)
    # But we can also use a different structure
    # Let's try: U[i,j] = 1 for all j, V[j,k] = 1 for all k, W[k,i] = 1
    # This is rank 5
    for k in range(5):
        U_col = [0.0] * (n * m)
        V_col = [0.0] * (m * p)
        W_col = [0.0] * (p * n)
        U_col[2*m + 0] = 1.0
        U_col[2*m + 1] = 1.0
        V_col[0*p + k] = 1.0
        V_col[1*p + k] = 1.0
        W_col[k*n + 2] = 1.0
        U_cols.append(U_col)
        V_cols.append(V_col)
        W_cols.append(W_col)

    # Block 4: i=2, j in {2,3} -> 1x2x5
    for k in range(5):
        U_col = [0.0] * (n * m)
        V_col = [0.0] * (m * p)
        W_col = [0.0] * (p * n)
        U_col[2*m + 2] = 1.0
        U_col[2*m + 3] = 1.0
        V_col[2*p + k] = 1.0
        V_col[3*p + k] = 1.0
        W_col[k*n + 2] = 1.0
        U_cols.append(U_col)
        V_cols.append(V_col)
        W_cols.append(W_col)

    rank = len(U_cols)
    U_mat = [[U_cols[c][r] for c in range(rank)] for r in range(n*m)]
    V_mat = [[V_cols[c][r] for c in range(rank)] for r in range(m*p)]
    W_mat = [[W_cols[c][r] for c in range(rank)] for r in range(p*n)]

    if verify_fast(U_mat, V_mat, W_mat, n, m, p):
        if rank < best_rank:
            best_result = {"U": U_mat, "V": V_mat, "W": W_mat}
            best_rank = rank

    return best_result
# EVOLVE-BLOCK-END
