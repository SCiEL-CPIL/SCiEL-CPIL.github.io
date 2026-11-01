"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    # 7 blocks of M_{2,2,2} (Strassen, rank 7) + 1 block of M_{2,2,1} (schoolbook, rank 4)
    # Total rank: 7*7 + 4 = 53
    rank = 53
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for M_{2,2,2}
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
    
    # Schoolbook coefficients for M_{2,2,1}
    U_b = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    V_b = [[1,0,1,0],[0,1,0,1]]
    W_b = [[1,1,0,0],[0,0,1,1]]
    
    col_offset = 0
    k_start = 0
    
    # 7 Strassen blocks for k = 0..13
    for block in range(7):
        k0 = k_start
        k1 = k_start + 1
        for r in range(7):
            c = col_offset + r
            for i in range(2):
                for j in range(2):
                    U[i*m+j][c] = U_s[i*2+j][r]
            for j in range(2):
                V[j*p+k0][c] = V_s[j*2+0][r]
                V[j*p+k1][c] = V_s[j*2+1][r]
            for kk in range(2):
                for i in range(2):
                    W[(k0+kk)*n+i][c] = W_s[kk*2+i][r]
        col_offset += 7
        k_start += 2
    
    # 1 schoolbook block for k = 14
    k_last = k_start
    for r in range(4):
        c = col_offset + r
        for i in range(2):
            for j in range(2):
                U[i*m+j][c] = U_b[i*2+j][r]
        for j in range(2):
            V[j*p+k_last][c] = V_b[j][r]
        for i in range(2):
            W[k_last*n+i][c] = W_b[i][r]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
