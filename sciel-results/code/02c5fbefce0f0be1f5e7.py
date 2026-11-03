"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    _start_time = time.time()
    _deadline = _start_time + 110.0  # 150s kill, leave margin
    
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # Step 1: Define Strassen 2x2x2 decomposition (Rank 7)
    # U, V, W are 4x7 matrices
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
    # T[i,j,k] = sum_c U2[i*2+j, c] * V2[j*2+k, c] * W2[k*2+i, c] == 1
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
    # U4 is 16x49, V4 is 16x49, W4 is 16x49
    # U4[(i1*2+i2)*4 + (j1*2+j2), c1*7+c2] = U2[i1*2+j1, c1] * U2[i2*2+j2, c2]
    
    r4 = 49
    U4 = [[0]*r4 for _ in range(16)]
    V4 = [[0]*r4 for _ in range(16)]
    W4 = [[0]*r4 for _ in range(16)]
    
    for i1 in range(2):
        for j1 in range(2):
            for c1 in range(7):
                u1 = U2[i1*2+j1][c1]
                if u1 == 0: continue
                for i2 in range(2):
                    for j2 in range(2):
                        u2 = U2[i2*2+j2][c1] # Wait, c1 is for first tensor, c2 for second?
                        # No, Kronecker: U[(i1,i2), (j1,j2), (c1,c2)] = U1[i1,j1,c1] * U2[i2,j2,c2]
                        # So we need to iterate c1, c2.
                        
    # Correct Kronecker construction
    U4 = [[0]*r4 for _ in range(16)]
    V4 = [[0]*r4 for _ in range(16)]
    W4 = [[0]*r4 for _ in range(16)]
    
    for i1 in range(2):
        for j1 in range(2):
            for c1 in range(7):
                u1 = U2[i1*2+j1][c1]
                if u1 == 0: continue
                for i2 in range(2):
                    for j2 in range(2):
                        for c2 in range(7):
                            u2 = U2[i2*2+j2][c2]
                            if u2 == 0: continue
                            idx_u = (i1*2 + i2)*4 + (j1*2 + j2)
                            idx_c = c1*7 + c2
                            U4[idx_u][idx_c] = u1 * u2
                            
                            v1 = V2[j1*2 + (k1:=0)][c1] # Need k1, k2 for V
                            # V indices: (j1, k1) and (j2, k2)
                            # So we need to iterate k1, k2 for V and W too.
                            
    # Full Kronecker Loop
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
                                    
                                    # Only if all non-zero? No, sum is linear.
                                    # But for sparsity, we can skip zeros.
                                    # However, Kronecker product of tensors is:
                                    # T4[i1,i2, j1,j2, k1,k2, c1,c2] = T2[i1,j1,k1,c1] * T2[i2,j2,k2,c2]
                                    # This is not a single rank-1 term in the new space unless we treat (c1,c2) as one index.
                                    # Yes, it is.
                                    
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

    # Step 3: Extend to 4x4x9
    # U is 16x49 (same as U4)
    # V is 36x49
    # W is 36x49
    # For k in 0..3, use V4, W4.
    # For k in 4..8, copy k=0 slice.
    
    U = [row[:] for row in U4]
    V = [[0]*49 for _ in range(36)]
    W = [[0]*49 for _ in range(36)]
    
    for j in range(4):
        for k in range(4):
            for c in range(49):
                V[j*9+k][c] = V4[j*4+k][c]
                W[k*4+j][c] = W4[k*4+j][c] # Wait, W index is k*4+i. Here i is j? No.
                # W4 index is k*4+i.
                # In V4, index is j*4+k.
                # In W4, index is k*4+i.
                # So for W, we need to fill W[k*4+i].
                
    # Redo W filling carefully
    W = [[0]*49 for _ in range(36)]
    for i in range(4):
        for k in range(4):
            for c in range(49):
                W[k*4+i][c] = W4[k*4+i][c]
                
    # Fill k=4..8
    for j in range(4):
        for k in range(4, 9):
            for c in range(49):
                V[j*9+k][c] = V4[j*4+0][c] # Copy k=0 slice
                
    for i in range(4):
        for k in range(4, 9):
            for c in range(49):
                W[k*4+i][c] = W4[0*4+i][c] # Copy k=0 slice
                
    # Final Verification
    for i in range(4):
        for j in range(4):
            for k in range(9):
                s = 0
                for c in range(49):
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
