"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=16
    # Rank is 56, using 8 blocks of 2x2x2 Strassen decompositions.
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen vectors for 2x2x2
    # u_r, v_r, w_r for r in 0..6
    # u_r[i*m+j], v_r[j*p_local+k_local], w_r[k_local*n+i]
    # p_local=2, n=2
    u_vecs = [
        [1, 0, 0, 1], # P1
        [0, 0, 1, 1], # P2
        [1, 1, 0, 0], # P3
        [1, 0, 0, 0], # P4
        [0, 0, 0, 1], # P5
        [0, 1, 0, -1],# P6
        [1, 0, -1, 0] # P7
    ]
    v_vecs = [
        [1, 0, 0, 1], # P1
        [1, 0, 0, 0], # P2
        [0, 0, 0, 1], # P3
        [1, -1, 0, 0],# P4
        [0, 1, -1, 0],# P5
        [0, 0, 1, 1], # P6
        [1, 1, 0, 0]  # P7
    ]
    w_vecs = [
        [1, 0, 0, 1], # P1
        [0, 1, 0, -1],# P2
        [0, 0, 1, 1], # P3
        [1, 0, 0, 0], # P4
        [-1, 0, 1, 0],# P5
        [0, 1, 0, 0], # P6
        [-1, 0, 0, 1] # P7
    ]
    
    num_blocks = p // 2 # 8 blocks
    
    for b in range(num_blocks):
        for r in range(7):
            R = b * 7 + r
            u = u_vecs[r]
            v = v_vecs[r]
            w = w_vecs[r]
            
            # Fill U
            for i in range(n):
                for j in range(m):
                    U[i * m + j][R] = u[i * m + j]
            
            # Fill V
            # V index: j * p + k
            # k = 2*b + c
            for j in range(m):
                for c in range(2):
                    k = 2 * b + c
                    V[j * p + k][R] = v[j * 2 + c]
            
            # Fill W
            # W index: k * n + i
            # k = 2*b + c
            for i in range(n):
                for c in range(2):
                    k = 2 * b + c
                    W[k * n + i][R] = w[c * n + i]
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
