"""
Rewritten program for matrix multiplication 2,2,15 tensor rank minimization.
Uses a block decomposition strategy with 7 blocks of M_{2,2,2} (rank 7)
and 1 block of M_{2,2,1} (rank 4), totaling rank 53.
"""

import numpy as np

def construct():
    n, m, p = 2, 2, 15
    rank = 53
    
    # Initialize U, V, W as lists of lists of zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen decomposition for M_{2,2,2}
    # U_strassen: 4x7, V_strassen: 4x7, W_strassen: 4x7
    U_s = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    V_s = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    W_s = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    
    # Schoolbook decomposition for M_{2,2,1}
    # U_school: 4x4, V_school: 2x4, W_school: 2x4
    U_sc = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]
    V_sc = [
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ]
    W_sc = [
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    
    col = 0
    # 7 blocks of M_{2,2,2}
    for b in range(7):
        k0 = 2 * b
        k1 = 2 * b + 1
        for r in range(7):
            c = col + r
            # U
            for i in range(2):
                for j in range(2):
                    U[i*m + j][c] = U_s[i*2 + j][r]
            # V
            for j in range(2):
                V[j*p + k0][c] = V_s[j*2 + 0][r]
                V[j*p + k1][c] = V_s[j*2 + 1][r]
            # W
            for k in range(2):
                kk = k0 + k
                for i in range(2):
                    W[kk*n + i][c] = W_s[k*2 + i][r]
        col += 7
        
    # 1 block of M_{2,2,1} at k=14
    k_last = 14
    for r in range(4):
        c = col + r
        # U
        for i in range(2):
            for j in range(2):
                U[i*m + j][c] = U_sc[i*2 + j][r]
        # V
        for j in range(2):
            V[j*p + k_last][c] = V_sc[j][r]
        # W
        for i in range(2):
            W[k_last*n + i][c] = W_sc[i][r]
            
    return {"U": U, "V": V, "W": W}
