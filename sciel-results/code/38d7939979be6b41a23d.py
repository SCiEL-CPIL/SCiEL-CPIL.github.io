"""Improved program using Hybrid Strassen/Schoolbook Block Strategy."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    _start_time = time.time()
    _deadline = _start_time + 110.0
    
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # Step 1: Define Strassen 2x2x2 decomposition (Rank 7)
    # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    U2 = [
        [1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1],
        [0, 1, 1, 0, 1, 0, 0],
        [1, 1, 0, 1, 0, 1, 1]
    ]
    V2 = [
        [1, -1, 0, 1, -1, 0, 1],
        [1, 0, 1, 0, 0, 1, 0],
        [1, 0, 0, -1, 1, 0, 0],
        [1, 1, 1, 1, 0, 1, 1]
    ]
    W2 = [
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 1, 0, 0],
        [1, 1, 0, 1, 0, 1, 1]
    ]
    
    # Verify 2x2x2
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = 0
                for c in range(7):
                    s += U2[i*2+j][c] * V2[j*2+k][c] * W2[k*2+i][c]
                if s != 1:
                    # Fallback to schoolbook if Strassen is broken
                    rank = n * m * p
                    U = [[0] * rank for _ in range(n * m)]
                    V = [[0] * rank for _ in range(m * p)]
                    W = [[0] * rank for _ in range(p * n)]
                    col = 0
                    for ii in range(n):
                        for jj in range(m):
                            for kk in range(p):
                                U[ii*m+jj][col] = 1
                                V[jj*p+kk][col] = 1
                                W[kk*n+ii][col] = 1
                                col += 1
                    return {"U": U, "V": V, "W": W}

    # Step 2: Kronecker product to get 4x4x4 decomposition (Rank 49)
    r4 = 49
    U4 = [[0]*r4 for _ in range(16)]
    V4 = [[0]*r4 for _ in range(16)]
    W4 = [[0]*r4 for _ in range(16)]
    
    for i1 in range(2):
        for j1 in range(2):
            for k1 in range(2):
                for i2 in range(2):
                    for j2 in range(2):
                        for k2 in range(2):
                            for c1 in range(7):
                                for c2 in range(7):
                                    idx_c = c1*7 + c2
                                    
                                    u1 = U2[i1*2+j1][c1]
                                    u2 = U2[i2*2+j2][c2]
                                    v1 = V2[j1*2+k1][c1]
                                    v2 = V2[j2*2+k2][c2]
                                    w1 = W2[k1*2+i1][c1]
                                    w2 = W2[k2*2+i2][c2]
                                    
                                    idx_u = (i1*2+i2)*4 + (j1*2+j2)
                                    idx_v = (j1*2+j2)*4 + (k1*2+k2)
                                    idx_w = (k1*2+k2)*4 + (i1*2+i2)
                                    
                                    U4[idx_u][idx_c] += u1 * u2
                                    V4[idx_v][idx_c] += v1 * v2
                                    W4[idx_w][idx_c] += w1 * w2
                                    
    # Verify 4x4x4
    for i in range(4):
        for j in range(4):
            for k in range(4):
                s = 0
                for c in range(49):
                    s += U4[i*4+j][c] * V4[j*4+k][c] * W4[k*4+i][c]
                if s != 1:
                    # Fallback
                    rank = n * m * p
                    U = [[0] * rank for _ in range(n * m)]
                    V = [[0] * rank for _ in range(m * p)]
                    W = [[0] * rank for _ in range(p * n)]
                    col = 0
                    for ii in range(n):
                        for jj in range(m):
                            for kk in range(p):
                                U[ii*m+jj][col] = 1
                                V[jj*p+kk][col] = 1
                                W[kk*n+ii][col] = 1
                                col += 1
                    return {"U": U, "V": V, "W": W}

    # Step 3: Construct 4x4x9 decomposition
    # Strategy: Split B into B1 (4x4), B2 (4x4), B3 (4x1)
    # Rank = 49 (Strassen) + 49 (Strassen) + 16 (Schoolbook) = 114
    
    R = 114
    U = [[0]*R for _ in range(n*m)]  # 16 x 114
    V = [[0]*R for _ in range(m*p)]  # 36 x 114
    W = [[0]*R for _ in range(p*n)]  # 36 x 114
    
    # Block 1: Columns 0-3 of B (Strassen 4x4x4)
    # U[:, 0:49] = U4
    for i in range(16):
        for c in range(49):
            U[i][c] = U4[i][c]
            
    # V: Rows j*9+k. For k in 0..3, map to V4[j*4+k]
    for j in range(4):
        for k in range(4):
            for c in range(49):
                V[j*9+k][c] = V4[j*4+k][c]
                
    # W: Rows k*4+i. For k in 0..3, map to W4[k*4+i]
    for k in range(4):
        for i in range(4):
            for c in range(49):
                W[k*4+i][c] = W4[k*4+i][c]
                
    # Block 2: Columns 4-7 of B (Strassen 4x4x4)
    # U[:, 49:98] = U4
    for i in range(16):
        for c in range(49):
            U[i][49+c] = U4[i][c]
            
    # V: Rows j*9+k. For k in 4..7, map to V4[j*4+(k-4)]
    for j in range(4):
        for k in range(4, 8):
            for c in range(49):
                V[j*9+k][49+c] = V4[j*4+(k-4)][c]
                
    # W: Rows k*4+i. For k in 4..7, map to W4[(k-4)*4+i]
    for k in range(4, 8):
        for i in range(4):
            for c in range(49):
                W[k*4+i][49+c] = W4[(k-4)*4+i][c]
                
    # Block 3: Column 8 of B (Schoolbook 4x4x1)
    # Rank 16. c in 98..113.
    # U[i*4+j, 98 + i*4+j] = 1
    # V[j*9+8, 98 + i*4+j] = 1
    # W[8*4+i, 98 + i*4+j] = 1
    for i in range(4):
        for j in range(4):
            c_idx = 98 + i*4 + j
            U[i*4+j][c_idx] = 1
            V[j*9+8][c_idx] = 1
            W[8*4+i][c_idx] = 1
            
    # Final Verification
    for i in range(4):
        for j in range(4):
            for k in range(9):
                s = 0
                for c in range(R):
                    s += U[i*4+j][c] * V[j*9+k][c] * W[k*4+i][c]
                if s != 1:
                    # Fallback to schoolbook
                    rank = n * m * p
                    U_f = [[0] * rank for _ in range(n * m)]
                    V_f = [[0] * rank for _ in range(m * p)]
                    W_f = [[0] * rank for _ in range(p * n)]
                    col = 0
                    for ii in range(n):
                        for jj in range(m):
                            for kk in range(p):
                                U_f[ii*m+jj][col] = 1
                                V_f[jj*p+kk][col] = 1
                                W_f[kk*n+ii][col] = 1
                                col += 1
                    return {"U": U_f, "V": V_f, "W": W_f}
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
