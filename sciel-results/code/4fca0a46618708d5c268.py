import time
import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    deadline = time.time() + 140  # Leave 10s margin before 150s kill
    
    n, m, p = DIMENSIONS  # 2, 3, 10
    
    # Build the rank-55 Strassen-based decomposition
    # Strategy:
    # - Split the 10 columns of B into 5 pairs of 2
    # - For each pair, use 7 Strassen multiplications on the 2x2 block 
    #   (rows 0,1 of A and rows 0,1 of B for those columns)
    # - Use 10 schoolbook terms for row 2 of A
    # Total: 5*7 + 10 = 55
    
    rank = 55
    U = np.zeros((n * m, rank), dtype=float)  # (6, 55)
    V = np.zeros((m * p, rank), dtype=float)  # (30, 55)
    W = np.zeros((p * n, rank), dtype=float)  # (20, 55)
    
    col = 0
    
    # Index helpers
    # A indices: a[i][j] -> U[i*m + j, col] = U[i*3 + j, col]
    # B indices: b[j][k] -> V[j*p + k, col] = V[j*10 + k, col]
    # C indices: c[i][k] -> W[k*n + i, col] = W[k*2 + i, col]
    
    # A row 0: j=0,1,2 -> indices 0, 1, 2
    # A row 1: j=0,1,2 -> indices 3, 4, 5
    # A row 2: j=0,1,2 -> indices 6, 7, 8 (wait, n=2, so only rows 0,1)
    
    # Correction: n=2, so A is 2x3, rows are 0 and 1
    # m=3, so columns of A are 0,1,2
    # U has shape (n*m, rank) = (6, rank)
    # U[i*m + j, col] where i in {0,1}, j in {0,1,2}
    
    # B is 3x10, rows 0,1,2, cols 0..9
    # V has shape (m*p, rank) = (30, rank)
    # V[j*p + k, col] where j in {0,1,2}, k in {0..9}
    
    # C is 2x10, rows 0,1, cols 0..9
    # W has shape (p*n, rank) = (20, rank)
    # W[k*n + i, col] where k in {0..9}, i in {0,1}
    
    # For each pair of columns (k0, k1) where k0=2b, k1=2b+1
    for b in range(5):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # B indices for this pair
        b00 = 0 * p + k0  # B[0][k0]
        b01 = 0 * p + k1  # B[0][k1]
        b10 = 1 * p + k0  # B[1][k0]
        b11 = 1 * p + k1  # B[1][k1]
        
        # C indices for this pair
        c00 = k0 * n + 0  # C[0][k0]
        c01 = k1 * n + 0  # C[0][k1]
        c10 = k0 * n + 1  # C[1][k0]
        c11 = k1 * n + 1  # C[1][k1]
        
        # A indices (only rows 0 and 1, cols 0,1,2)
        a00 = 0 * m + 0  # A[0][0]
        a01 = 0 * m + 1  # A[0][1]
        a10 = 1 * m + 0  # A[1][0]
        a11 = 1 * m + 1  # A[1][1]
        
        # Strassen for 2x2: computing C = A * B where
        # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
        # C = [[c00, c01], [c10, c11]]
        #
        # But wait - A has 3 columns, not 2. We're only using columns 0 and 1
        # of A for this Strassen block. Column 2 of A will be handled separately.
        
        # M1 = (A[0][0] + A[1][1]) * (B[0][k0] + B[1][k1])
        U[a00, col] += 1.0
        U[a11, col] += 1.0
        V[b00, col] += 1.0
        V[b11, col] += 1.0
        W[c00, col] += 1.0
        W[c11, col] += 1.0
        col += 1
        
        # M2 = (A[1][0] + A[1][1]) * B[0][k0]
        U[a10, col] += 1.0
        U[a11, col] += 1.0
        V[b00, col] += 1.0
        W[c10, col] += 1.0
        W[c11, col] -= 1.0
        col += 1
        
        # M3 = A[0][0] * (B[0][k1] - B[1][k1])
        U[a00, col] += 1.0
        V[b01, col] += 1.0
        V[b11, col] -= 1.0
        W[c01, col] += 1.0
        W[c11, col] += 1.0
        col += 1
        
        # M4 = A[1][1] * (B[1][k0] - B[0][k0])
        U[a11, col] += 1.0
        V[b10, col] += 1.0
        V[b00, col] -= 1.0
        W[c00, col] += 1.0
        W[c10, col] += 1.0
        col += 1
        
        # M5 = (A[0][0] + A[0][1]) * B[1][k1]
        U[a00, col] += 1.0
        U[a01, col] += 1.0
        V[b11, col] += 1.0
        W[c00, col] -= 1.0
        W[c01, col] += 1.0
        col += 1
        
        # M6 = (A[1][0] - A[0][0]) * (B[0][k0] + B[0][k1])
        U[a10, col] += 1.0
        U[a00, col] -= 1.0
        V[b00, col] += 1.0
        V[b01, col] += 1.0
        W[c11, col] += 1.0
        col += 1
        
        # M7 = (A[0][1] - A[1][1]) * (B[1][k0] + B[1][k1])
        U[a01, col] += 1.0
        U[a11, col] -= 1.0
        V[b10, col] += 1.0
        V[b11, col] += 1.0
        W[c00, col] += 1.0
        col += 1
    
    # Now handle the third column of A (column 2)
    # A[0][2] and A[1][2] multiply all 10 columns of B
    a02 = 0 * m + 2  # A[0][2]
    a12 = 1 * m + 2  # A[1][2]
    
    for k in range(p):
        b0k = 0 * p + k  # B[0][k]
        b1k = 1 * p + k  # B[1][k]
        b2k = 2 * p + k  # B[2][k]
        
        c0k = k * n + 0  # C[0][k]
        c1k = k * n + 1  # C[1][k]
        
        # A[0][2] * B[0][k] + A[0][2] * B[1][k] + A[0][2] * B[2][k]
        # Wait, no. C[0][k] = sum_j A[0][j] * B[j][k]
        # We've already handled j=0 and j=1 via Strassen for pairs.
        # Now we need j=2: A[0][2] * B[2][k]
        
        U[a02, col] += 1.0
        V[b2k, col] += 1.0
        W[c0k, col] += 1.0
        col += 1
        
        U[a12, col] += 1.0
        V[b2k, col] += 1.0
        W[c1k, col] += 1.0
        col += 1
    
    # Verify the decomposition
    def verify(U_mat, V_mat, W_mat, r):
        n_v, m_v, p_v = 2, 3, 10
        for i in range(n_v):
            for j in range(m_v):
                for k in range(p_v):
                    s = 0.0
                    for c in range(r):
                        s += U_mat[i * m_v + j, c] * V_mat[j * p_v + k, c] * W_mat[k * n_v + i, c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    if not verify(U, V, W, rank):
        # Fall back to schoolbook if verification fails
        rank_sb = n * m * p
        U_sb = np.zeros((n * m, rank_sb), dtype=float)
        V_sb = np.zeros((m * p, rank_sb), dtype=float)
        W_sb = np.zeros((p * n, rank_sb), dtype=float)
        col_sb = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j, col_sb] = 1
                    V_sb[j * p + k, col_sb] = 1
                    W_sb[k * n + i, col_sb] = 1
                    col_sb += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
