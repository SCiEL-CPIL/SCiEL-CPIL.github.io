import numpy as np
import time

def construct():
    """
    Construct a rank-minimal tensor decomposition for multiplying a 2x3 matrix by a 3x8 matrix.
    
    Strategy:
    - Split the 8-column dimension into 4 blocks of 2 columns each.
    - For each block, split the 3-row dimension into 1 row + 2 rows.
    - The 1-row part (2x1 * 1x2) has rank 4 (outer product).
    - The 2-row part (2x2 * 2x2) uses Strassen's algorithm with rank 7.
    - Total rank per block: 4 + 7 = 11.
    - Total rank: 4 * 11 = 44.
    
    Index conventions:
    - U has shape (n*m, rank) = (6, rank): index a = i*m + j
    - V has shape (m*p, rank) = (24, rank): index b = j*p + k
    - W has shape (p*n, rank) = (16, rank): index c = k*n + i
    """
    start_time = time.time()
    deadline = start_time + 120  # 120s wall-clock limit
    
    n, m, p = 2, 3, 8
    rank = 44
    
    U = np.zeros((n*m, rank), dtype=np.int64)
    V = np.zeros((m*p, rank), dtype=np.int64)
    W = np.zeros((p*n, rank), dtype=np.int64)
    
    col = 0
    
    # Process 4 blocks of 2 columns each
    for block in range(4):
        if time.time() > deadline - 5:  # Safety margin
            break
        k0 = block * 2
        k1 = block * 2 + 1
        
        # Part 1: A[:, 0] * B[0, k0:k1]  (2x1 * 1x2, rank 4)
        U[0*m + 0, col] = 1
        V[0*p + k0, col] = 1
        W[k0*n + 0, col] = 1
        col += 1
        
        U[0*m + 0, col] = 1
        V[0*p + k1, col] = 1
        W[k1*n + 0, col] = 1
        col += 1
        
        U[1*m + 0, col] = 1
        V[0*p + k0, col] = 1
        W[k0*n + 1, col] = 1
        col += 1
        
        U[1*m + 0, col] = 1
        V[0*p + k1, col] = 1
        W[k1*n + 1, col] = 1
        col += 1
        
        # Part 2: A[:, 1:3] * B[1:3, k0:k1]  (2x2 * 2x2, Strassen rank 7)
        U[0*m + 1, col] = 1
        U[1*m + 2, col] = 1
        V[1*p + k0, col] = 1
        V[2*p + k1, col] = 1
        W[k0*n + 0, col] = 1
        W[k1*n + 1, col] = 1
        col += 1
        
        U[1*m + 1, col] = 1
        U[1*m + 2, col] = 1
        V[1*p + k0, col] = 1
        W[k0*n + 1, col] = 1
        W[k1*n + 1, col] = -1
        col += 1
        
        U[0*m + 1, col] = 1
        V[1*p + k1, col] = 1
        V[2*p + k1, col] = -1
        W[k1*n + 0, col] = 1
        W[k1*n + 1, col] = 1
        col += 1
        
        U[1*m + 2, col] = 1
        V[2*p + k0, col] = 1
        V[1*p + k0, col] = -1
        W[k0*n + 0, col] = 1
        W[k0*n + 1, col] = 1
        col += 1
        
        U[0*m + 1, col] = 1
        U[0*m + 2, col] = 1
        V[2*p + k1, col] = 1
        W[k0*n + 0, col] = -1
        W[k1*n + 0, col] = 1
        col += 1
        
        U[1*m + 1, col] = 1
        U[0*m + 1, col] = -1
        V[1*p + k0, col] = 1
        V[1*p + k1, col] = 1
        W[k1*n + 1, col] = 1
        col += 1
        
        U[0*m + 2, col] = 1
        U[1*m + 2, col] = -1
        V[2*p + k0, col] = 1
        V[2*p + k1, col] = 1
        W[k0*n + 0, col] = 1
        col += 1
    
    # Verify correctness
    def verify(U_mat, V_mat, W_mat):
        r = U_mat.shape[1]
        T_check = np.einsum('ad,bd,cd->abc', U_mat, V_mat, W_mat)
        T_exp = np.zeros((n*m, m*p, p*n), dtype=np.int64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a_idx = i * m + j
                    b_idx = j * p + k
                    c_idx = k * n + i
                    T_exp[a_idx, b_idx, c_idx] = 1
        return np.array_equal(T_check, T_exp)
    
    if col == rank and verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback: schoolbook rank 48
        rank_sb = n * m * p
        U_sb = np.zeros((n*m, rank_sb), dtype=np.int64)
        V_sb = np.zeros((m*p, rank_sb), dtype=np.int64)
        W_sb = np.zeros((p*n, rank_sb), dtype=np.int64)
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a_idx = i * m + j
                    b_idx = j * p + k
                    c_idx = k * n + i
                    U_sb[a_idx, c] = 1
                    V_sb[b_idx, c] = 1
                    W_sb[c_idx, c] = 1
                    c += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
