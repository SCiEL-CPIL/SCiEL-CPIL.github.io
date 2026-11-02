import numpy as np
import time

def construct():
    n, m, p = 2, 4, 5
    t0 = time.time()
    deadline = t0 + 120  # 120 s wall-clock budget with margin

    # We use the same block structure as the current program:
    # 4 Strassen 2x2x2 blocks (rank 7 each) + 2 schoolbook 2x2x1 blocks (rank 4 each)
    # Total rank = 4*7 + 2*4 = 28 + 8 = 36

    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def add_strassen(col_offset, j_offset, k_offset):
        # j_offset: 0 for J1, 2 for J2
        # k_offset: 0 for K1, 2 for K2
        # Strassen for 2x2x2
        # U indices: i*m + (j_offset + j')
        # V indices: (j_offset + j')*p + (k_offset + k')
        # W indices: (k_offset + k')*n + i

        # Term 1: M1 = (A00 + A11) * (B00 + B11)
        U[0 * m + j_offset + 0][col_offset + 0] += 1
        U[1 * m + j_offset + 1][col_offset + 0] += 1
        V[(j_offset + 0) * p + k_offset + 0][col_offset + 0] += 1
        V[(j_offset + 1) * p + k_offset + 1][col_offset + 0] += 1
        W[(k_offset + 0) * n + 0][col_offset + 0] += 1
        W[(k_offset + 1) * n + 1][col_offset + 0] += 1

        # Term 2: M2 = (A10 + A11) * B00
        U[1 * m + j_offset + 0][col_offset + 1] += 1
        U[1 * m + j_offset + 1][col_offset + 1] += 1
        V[(j_offset + 0) * p + k_offset + 0][col_offset + 1] += 1
        W[(k_offset + 0) * n + 1][col_offset + 1] += 1
        W[(k_offset + 1) * n + 1][col_offset + 1] += -1

        # Term 3: M3 = A00 * (B01 - B11)
        U[0 * m + j_offset + 0][col_offset + 2] += 1
        V[(j_offset + 0) * p + k_offset + 1][col_offset + 2] += 1
        V[(j_offset + 1) * p + k_offset + 1][col_offset + 2] += -1
        W[(k_offset + 1) * n + 0][col_offset + 2] += 1
        W[(k_offset + 1) * n + 1][col_offset + 2] += 1

        # Term 4: M4 = A11 * (B10 - B00)
        U[1 * m + j_offset + 1][col_offset + 3] += 1
        V[(j_offset + 0) * p + k_offset + 0][col_offset + 3] += -1
        V[(j_offset + 1) * p + k_offset + 0][col_offset + 3] += 1
        W[(k_offset + 0) * n + 0][col_offset + 3] += 1
        W[(k_offset + 0) * n + 1][col_offset + 3] += 1

        # Term 5: M5 = (A00 + A01) * B11
        U[0 * m + j_offset + 0][col_offset + 4] += 1
        U[0 * m + j_offset + 1][col_offset + 4] += 1
        V[(j_offset + 1) * p + k_offset + 1][col_offset + 4] += 1
        W[(k_offset + 0) * n + 0][col_offset + 4] += -1
        W[(k_offset + 1) * n + 0][col_offset + 4] += 1

        # Term 6: M6 = (A10 - A00) * (B00 + B01)
        U[0 * m + j_offset + 0][col_offset + 5] += -1
        U[1 * m + j_offset + 0][col_offset + 5] += 1
        V[(j_offset + 0) * p + k_offset + 0][col_offset + 5] += 1
        V[(j_offset + 0) * p + k_offset + 1][col_offset + 5] += 1
        W[(k_offset + 1) * n + 1][col_offset + 5] += 1

        # Term 7: M7 = (A01 - A11) * (B10 + B11)
        U[0 * m + j_offset + 1][col_offset + 6] += 1
        U[1 * m + j_offset + 1][col_offset + 6] += -1
        V[(j_offset + 1) * p + k_offset + 0][col_offset + 6] += 1
        V[(j_offset + 1) * p + k_offset + 1][col_offset + 6] += 1
        W[(k_offset + 0) * n + 0][col_offset + 6] += 1

    def add_schoolbook_2x2x1(col_offset, j_offset, k_offset):
        # Schoolbook for 2x2x1
        # k is fixed to k_offset
        for di in range(2):
            for dj in range(2):
                U[di * m + j_offset + dj][col_offset + di*2 + dj] = 1
                V[(j_offset + dj) * p + k_offset][col_offset + di*2 + dj] = 1
                W[k_offset * n + di][col_offset + di*2 + dj] = 1

    col = 0
    # J1 (j_offset=0)
    add_strassen(col, 0, 0); col += 7
    add_strassen(col, 0, 2); col += 7
    add_schoolbook_2x2x1(col, 0, 4); col += 4

    # J2 (j_offset=2)
    add_strassen(col, 2, 0); col += 7
    add_strassen(col, 2, 2); col += 7
    add_schoolbook_2x2x1(col, 2, 4); col += 4

    # Verify correctness with random test
    if time.time() < deadline - 5:
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Random matrices
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        
        # Expected product
        AB = A @ B
        
        # Computed via tensor decomposition
        # C[i,k] = sum_j A[i,j] B[j,k]
        # Using decomposition:
        # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        # U is indexed by (i*m + j, r)
        # V is indexed by (j*p + k, r)
        # W is indexed by (k*n + i, r)
        
        # Reshape A to (n*m,) and B to (m*p,)
        A_flat = A.flatten()  # i*m + j order
        B_flat = B.flatten()  # j*p + k order
        
        # Compute intermediate products
        prod = U_np.T @ A_flat  # (rank,)
        prod = prod * (V_np.T @ B_flat)  # element-wise, (rank,)
        C_flat = W_np.T @ prod  # (p*n,)
        
        C = C_flat.reshape(p, n).T  # back to (n, p)
        
        if not np.allclose(AB, C, atol=1e-10):
            # Fallback: return schoolbook (rank 40)
            rank_sb = n * m * p
            U_sb = [[0] * rank_sb for _ in range(n * m)]
            V_sb = [[0] * rank_sb for _ in range(m * p)]
            W_sb = [[0] * rank_sb for _ in range(p * n)]
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_sb[i*m + j][idx] = 1
                        V_sb[j*p + k][idx] = 1
                        W_sb[k*n + i][idx] = 1
                        idx += 1
            return {"U": U_sb, "V": V_sb, "W": W_sb}

    return {"U": U, "V": V, "W": W}
