import numpy as np
import time

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=5
    # Rank 49 for 4x4x4, Rank 16 for 4x4x1. Total 65.
    rank = 65
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # 2x2 Strassen decomposition vectors
    # Order of entries: 00, 01, 10, 11
    # U (A), V (B), W (C)
    
    # t=0: m1 = (a+d)(e+h)
    u0 = [1, 0, 0, 1]
    v0 = [1, 0, 0, 1]
    w0 = [1, 1, 1, 1] # p,q,r,s -> 1,1,1,1
    
    # t=1: m2 = (c+d)e
    u1 = [0, 0, 1, 1]
    v1 = [1, 0, 0, 0]
    w1 = [-1, 0, 1, 0] # p=-1, r=1
    
    # t=2: m3 = a(f-h)
    u2 = [1, 0, 0, 0]
    v2 = [0, 1, 0, -1]
    w2 = [0, 1, 0, -1] # q=1, s=-1
    
    # t=3: m4 = d(g-e)
    u3 = [0, 0, 0, 1]
    v3 = [-1, 0, 1, 0]
    w3 = [1, 0, 0, 1] # p=1, s=1
    
    # t=4: m5 = (a+b)h
    u4 = [1, 1, 0, 0]
    v4 = [0, 0, 0, 1]
    w4 = [0, 0, 0, 1] # s=1
    
    # t=5: m6 = (c-a)(e+f)
    u5 = [-1, 0, 1, 0]
    v5 = [1, 1, 0, 0]
    w5 = [0, 0, 1, 0] # r=1
    
    # t=6: m7 = (b-d)(g+h)
    u6 = [0, 1, 0, -1]
    v6 = [0, 0, 1, 1]
    w6 = [-1, 1, 0, 0] # p=-1, q=1
    
    U_2x2 = [u0, u1, u2, u3, u4, u5, u6]
    V_2x2 = [v0, v1, v2, v3, v4, v5, v6]
    W_2x2 = [w0, w1, w2, w3, w4, w5, w6]
    
    # Generate 49 terms for 4x4x4
    col = 0
    for s in range(7):
        for t in range(7):
            # Kronecker product
            u_kron = np.kron(U_2x2[s], U_2x2[t])
            v_kron = np.kron(V_2x2[s], V_2x2[t])
            w_kron = np.kron(W_2x2[s], W_2x2[t])
            
            # Fill U (length 16)
            for idx in range(16):
                U[idx][col] = u_kron[idx]
                
            # Fill V (length 20, first 16 are 4x4 part)
            for idx in range(16):
                V[idx][col] = v_kron[idx]
            # Last 4 are 0 (already initialized)
            
            # Fill W (length 20, first 16 are 4x4 part)
            for idx in range(16):
                W[idx][col] = w_kron[idx]
            # Last 4 are 0
            
            col += 1
            
    # Generate 16 terms for 4x4x1 (5th column)
    for i in range(n):
        for j in range(m):
            # U: index i*m + j
            U[i * m + j][col] = 1.0
            
            # V: index j*p + k, k=4
            V[j * p + 4][col] = 1.0
            
            # W: index k*n + i, k=4
            W[4 * n + i][col] = 1.0
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
