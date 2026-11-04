"""Rewritten program for matrix multiplication 3x4x5 tensor decomposition.
Uses a subspace/block decomposition approach with Strassen on 2x2x2 blocks
and schoolbook for the remaining parts, aiming for rank 56.
"""
import time
import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 5
    deadline = time.time() + 115.0

    def strassen_2x2x2():
        """Return U, V, W for 2x2x2 Strassen decomposition (rank 7)."""
        # Strassen's 7 products for 2x2 matrix multiplication
        # U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
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

    def verify(U, V, W, n, m, p, tol=1e-9):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        rank = U_arr.shape[1]
        U_r = U_arr.reshape(n, m, rank)
        V_r = V_arr.reshape(m, p, rank)
        W_r = W_arr.reshape(p, n, rank)
        T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r, optimize=True)
        return np.allclose(T, np.ones((n, m, p)), atol=tol, rtol=0)

    U_cols = []
    V_cols = []
    W_cols = []

    U_s, V_s, W_s = strassen_2x2x2()

    # Partition: i in {0,1} use Strassen blocks, i=2 uses schoolbook
    # For i in {0,1}:
    #   j in {0,1}, k in {0,1}: Strassen (7 terms)
    #   j in {2,3}, k in {0,1}: Strassen (7 terms)
    #   j in {0,1}, k in {2,3}: Strassen (7 terms)
    #   j in {2,3}, k in {2,3}: Strassen (7 terms)
    #   k=4: schoolbook for all j (4 terms)
    # For i=2: schoolbook for all j,k (20 terms)
    # Total: 4*7 + 4 + 20 = 28 + 4 + 20 = 52

    # Block 1: i in {0,1}, j in {0,1}, k in {0,1}
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

    # Block 2: i in {0,1}, j in {2,3}, k in {0,1}
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

    # Block 3: i in {0,1}, j in {0,1}, k in {2,3}
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

    # Block 4: i in {0,1}, j in {2,3}, k in {2,3}
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

    # Block 5: i in {0,1}, all j, k=4: schoolbook (4 terms)
    for i in range(2):
        for j in range(m):
            U_col = [0.0] * (n * m)
            V_col = [0.0] * (m * p)
            W_col = [0.0] * (p * n)
            U_col[i*m + j] = 1.0
            V_col[j*p + 4] = 1.0
            W_col[4*n + i] = 1.0
            U_cols.append(U_col)
            V_cols.append(V_col)
            W_cols.append(W_col)

    # Block 6: i=2, all j, k: schoolbook (20 terms)
    for j in range(m):
        for k in range(p):
            U_col = [0.0] * (n * m)
            V_col = [0.0] * (m * p)
            W_col = [0.0] * (p * n)
            U_col[2*m + j] = 1.0
            V_col[j*p + k] = 1.0
            W_col[k*n + 2] = 1.0
            U_cols.append(U_col)
            V_cols.append(V_col)
            W_cols.append(W_col)

    rank = len(U_cols)
    U_mat = [[U_cols[c][r] for c in range(rank)] for r in range(n*m)]
    V_mat = [[V_cols[c][r] for c in range(rank)] for r in range(m*p)]
    W_mat = [[W_cols[c][r] for c in range(rank)] for r in range(p*n)]

    best_result = {"U": U_mat, "V": V_mat, "W": W_mat}
    best_rank = rank

    if verify(U_mat, V_mat, W_mat, n, m, p):
        pass
    else:
        # Fallback to schoolbook if verification fails
        schoolbook_rank = n * m * p
        U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
        V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
        W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m+j][col] = 1
                    V_sb[j*p+k][col] = 1
                    W_sb[k*n+i][col] = 1
                    col += 1
        best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
        best_rank = schoolbook_rank

    return best_result
# EVOLVE-BLOCK-END
