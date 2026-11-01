"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS  # n=2, m=5, p=5
    
    # Total rank calculation
    # Blocks:
    # J partitions: [0,1], [2,3], [4]
    # K partitions: [0,1], [2,3], [4]
    # 4 blocks of 2x2x2 (Strassen, rank 7)
    # 4 blocks of 2x1x2 or 2x2x1 (Schoolbook, rank 4)
    # 1 block of 2x1x1 (Schoolbook, rank 2)
    # Total rank = 4*7 + 4*4 + 1*2 = 28 + 16 + 2 = 46
    
    total_rank = 46
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col = 0
    
    # Define partitions
    J_parts = [(0, 2), (2, 4), (4, 5)]
    K_parts = [(0, 2), (2, 4), (4, 5)]
    
    # Strassen vectors for 2x2x2
    # u: [a, b, c, d]
    # v: [e, f, g, h]
    # w: [C00, C01, C10, C11]
    strassen_u = [
        [1, 0, 0, 1], # M1: a+d
        [0, 0, 1, 1], # M2: c+d
        [1, 0, 0, 0], # M3: a
        [0, 0, 0, 1], # M4: d
        [1, 1, 0, 0], # M5: a+b
        [-1, 0, 1, 0],# M6: c-a
        [0, 1, 0, -1] # M7: b-d
    ]
    strassen_v = [
        [1, 1, 0, 0], # M1: e+h
        [1, 0, 0, 0], # M2: e
        [0, 1, 0, -1],# M3: f-h
        [0, 0, 1, -1],# M4: g-e
        [0, 0, 0, 1], # M5: h
        [1, 1, 0, 0], # M6: e+f
        [0, 0, 1, 1]  # M7: g+h
    ]
    strassen_w = [
        [1, 0, 0, 1], # M1: C00+C11
        [0, 0, 1, -1],# M2: C10-C11
        [0, 1, 0, 1], # M3: C01+C11
        [1, 0, 1, 0], # M4: C00+C10
        [-1, 1, 0, 0],# M5: -C00+C01
        [0, 0, 0, 1], # M6: C11
        [1, 0, 0, 0]  # M7: C00
    ]
    
    for j_start, j_end in J_parts:
        for k_start, k_end in K_parts:
            j_size = j_end - j_start
            k_size = k_end - k_start
            
            if j_size == 2 and k_size == 2:
                # Strassen block
                for t in range(7):
                    u_vec = strassen_u[t]
                    v_vec = strassen_v[t]
                    w_vec = strassen_w[t]
                    
                    # Fill U
                    # Local i in 0..1, j in 0..1
                    # Global row = i*m + (j_start + j_local)
                    for i in range(2):
                        for j_loc in range(2):
                            row = i * m + (j_start + j_loc)
                            U[row, col] = u_vec[i * 2 + j_loc]
                            
                    # Fill V
                    # Local j in 0..1, k in 0..1
                    # Global row = (j_start + j_loc)*p + (k_start + k_loc)
                    for j_loc in range(2):
                        for k_loc in range(2):
                            row = (j_start + j_loc) * p + (k_start + k_loc)
                            V[row, col] = v_vec[j_loc * 2 + k_loc]
                            
                    # Fill W
                    # Local k in 0..1, i in 0..1
                    # Global row = (k_start + k_loc)*n + i
                    for k_loc in range(2):
                        for i in range(2):
                            row = (k_start + k_loc) * n + i
                            W[row, col] = w_vec[k_loc * 2 + i]
                            
                    col += 1
            else:
                # Schoolbook block
                # Rank = 2 * j_size * k_size
                for i in range(2):
                    for j_loc in range(j_size):
                        for k_loc in range(k_size):
                            row_u = i * m + (j_start + j_loc)
                            row_v = (j_start + j_loc) * p + (k_start + k_loc)
                            row_w = (k_start + k_loc) * n + i
                            
                            U[row_u, col] = 1
                            V[row_v, col] = 1
                            W[row_w, col] = 1
                            
                            col += 1
                            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
