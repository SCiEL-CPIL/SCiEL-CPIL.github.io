import numpy as np
import time

DIMENSIONS = [6, 6, 6]

def construct():
    n, m, p = DIMENSIONS
    # Rank 189: 3x3 block decomposition with 2x2 Strassen
    rank = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2
    # Local indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    A_coeffs = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0],# M6
        [0, 1, 0, -1] # M7
    ]
    B_coeffs = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1],# M3
        [-1, 0, 1, 0],# M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    
    # Reconstruction coefficients for C_out
    # C_out indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    W_coeffs = [
        [1, 0, 0, 1], # M1 -> C00, C11
        [0, 0, 1, -1],# M2 -> C10, C11
        [0, 1, 0, 1], # M3 -> C01, C11
        [1, 0, 1, 0], # M4 -> C00, C10
        [-1, 1, 0, 0],# M5 -> C00, C01
        [0, 0, 0, 1], # M6 -> C11
        [1, 0, 0, 0]  # M7 -> C00
    ]
    
    col = 0
    for I in range(3):
        for K in range(3):
            for J in range(3):
                for s in range(7):
                    # A Block (I, J)
                    # B Block (J, K)
                    # C Block (I, K)
                    
                    for l_a in range(4):
                        val_a = A_coeffs[s][l_a]
                        if val_a != 0:
                            r_a = l_a // 2
                            c_a = l_a % 2
                            g_r_a = I * 2 + r_a
                            g_c_a = J * 2 + c_a
                            idx_a = g_r_a * 6 + g_c_a
                            U[idx_a][col] += val_a
                            
                    for l_b in range(4):
                        val_b = B_coeffs[s][l_b]
                        if val_b != 0:
                            r_b = l_b // 2
                            c_b = l_b % 2
                            g_r_b = J * 2 + r_b
                            g_c_b = K * 2 + c_b
                            idx_b = g_r_b * 6 + g_c_b
                            V[idx_b][col] += val_b
                            
                    for l_c in range(4):
                        val_c = W_coeffs[s][l_c]
                        if val_c != 0:
                            r_c = l_c // 2
                            c_c = l_c % 2
                            g_r_c = I * 2 + r_c
                            g_c_c = K * 2 + c_c
                            # W index: k * n + i
                            idx_w = g_c_c * 6 + g_r_c
                            W[idx_w][col] += val_c
                            
                    col += 1
                    
    return {"U": U, "V": V, "W": W}
