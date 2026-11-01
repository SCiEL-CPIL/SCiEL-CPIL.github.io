"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    A_dim = n * m  # 8
    B_dim = m * p  # 32
    C_dim = p * n  # 16
    
    # Rank is 56
    rank = 56
    U = np.zeros((A_dim, rank), dtype=float)
    V = np.zeros((B_dim, rank), dtype=float)
    W = np.zeros((C_dim, rank), dtype=float)
    
    # Strassen decomposition for 2x2x2
    # Local indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # U, V, W vectors for the 7 products
    strassen_U = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 0, 1, 0], # M5
        [-1, 0, 0, 1],# M6
        [0, 1, 0, -1] # M7
    ]
    strassen_V = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 0, 1, -1],# M3
        [-1, 1, 0, 0],# M4
        [0, 0, 0, 1], # M5
        [1, 0, 1, 0], # M6
        [0, 1, 0, 1]  # M7
    ]
    strassen_W = [
        [1, 0, 0, 1], # M1 -> Z00, Z11
        [0, 0, 1, -1],# M2 -> Z10, Z11
        [0, 1, 0, 1], # M3 -> Z01, Z11
        [1, 0, 1, 0], # M4 -> Z00, Z10
        [-1, 1, 0, 0],# M5 -> Z00, Z01
        [0, 0, 0, 1], # M6 -> Z11
        [1, 0, 0, 0]  # M7 -> Z00
    ]
    
    col_idx = 0
    for q in range(4):
        for s in range(2):
            for t in range(7):
                u_vec = strassen_U[t]
                v_vec = strassen_V[t]
                w_vec = strassen_W[t]
                
                for l in range(4):
                    if u_vec[l] != 0:
                        i = l // 2
                        jp = l % 2
                        a = i * m + s * 2 + jp
                        U[a, col_idx] += u_vec[l]
                        
                    if v_vec[l] != 0:
                        jp = l // 2
                        kp = l % 2
                        b = (s * 2 + jp) * p + 2 * q + kp
                        V[b, col_idx] += v_vec[l]
                        
                    if w_vec[l] != 0:
                        kp = l // 2
                        i = l % 2
                        c = (2 * q + kp) * n + i
                        W[c, col_idx] += w_vec[l]
                        
                col_idx += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
