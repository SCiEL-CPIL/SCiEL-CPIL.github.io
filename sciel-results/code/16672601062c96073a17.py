import numpy as np
import time

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    rank = 42
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    steps = [
        (
            {(0,0): 1, (0,1): 1, (1,2): 1, (1,3): 1},
            {(0,0): 1, (0,1): 1, (0,2): 1, (1,0): 1, (1,1): 1, (1,2): 1, (2,3): 1, (2,4): 1, (2,5): 1, (3,3): 1, (3,4): 1, (3,5): 1},
            [(0, 1), (3, 1)]
        ),
        (
            {(1,0): 1, (1,1): 1, (1,2): 1, (1,3): 1},
            {(0,0): 1, (0,1): 1, (0,2): 1, (1,0): 1, (1,1): 1, (1,2): 1},
            [(2, 1), (3, -1)]
        ),
        (
            {(0,0): 1, (0,1): 1},
            {(0,3): 1, (0,4): 1, (0,5): 1, (1,3): 1, (1,4): 1, (1,5): 1, (2,3): -1, (2,4): -1, (2,5): -1, (3,3): -1, (3,4): -1, (3,5): -1},
            [(1, 1), (3, 1)]
        ),
        (
            {(1,2): 1, (1,3): 1},
            {(2,0): 1, (2,1): 1, (2,2): 1, (3,0): 1, (3,1): 1, (3,2): 1, (0,0): -1, (0,1): -1, (0,2): -1, (1,0): -1, (1,1): -1, (1,2): -1},
            [(0, 1), (2, 1)]
        ),
        (
            {(0,0): 1, (0,1): 1, (0,2): 1, (0,3): 1},
            {(2,3): 1, (2,4): 1, (2,5): 1, (3,3): 1, (3,4): 1, (3,5): 1},
            [(0, -1), (1, 1)]
        ),
        (
            {(1,0): 1, (1,1): 1, (0,0): -1, (0,1): -1},
            {(0,0): 1, (0,1): 1, (0,2): 1, (0,3): 1, (0,4): 1, (0,5): 1, (1,0): 1, (1,1): 1, (1,2): 1, (1,3): 1, (1,4): 1, (1,5): 1},
            [(3, 1)]
        ),
        (
            {(0,2): 1, (0,3): 1, (1,2): -1, (1,3): -1},
            {(2,0): 1, (2,1): 1, (2,2): 1, (2,3): 1, (2,4): 1, (2,5): 1, (3,0): 1, (3,1): 1, (3,2): 1, (3,3): 1, (3,4): 1, (3,5): 1},
            [(0, 1)]
        ),
    ]
    
    col = 0
    for X_dict, Y_dict, C_targets in steps:
        x0_coeffs = {}
        x1_coeffs = {}
        for (i, j), c in X_dict.items():
            if j % 2 == 0:
                x0_coeffs[(i, j)] = c
            else:
                x1_coeffs[(i, j)] = c
                
        y_coeffs = [[], [], [], [], [], []]
        for (j, k), c in Y_dict.items():
            r_block = j % 2
            c_block = k % 3
            idx = r_block * 3 + c_block
            y_coeffs[idx].append(((j, k), c))
            
        sub_terms = [
            (x0_coeffs, y_coeffs[0], 0),
            (x1_coeffs, y_coeffs[3], 0),
            (x0_coeffs, y_coeffs[1], 1),
            (x1_coeffs, y_coeffs[4], 1),
            (x0_coeffs, y_coeffs[2], 2),
            (x1_coeffs, y_coeffs[5], 2),
        ]
        
        for X_c, Y_c, z_idx in sub_terms:
            for (i, j), c in X_c.items():
                U[i * m + j, col] += c
            for (j, k), c in Y_c:
                V[j * p + k, col] += c
            
            for block_idx, sign in C_targets:
                if block_idx == 0:
                    i_C = 0
                    k_C = z_idx
                elif block_idx == 1:
                    i_C = 0
                    k_C = 3 + z_idx
                elif block_idx == 2:
                    i_C = 1
                    k_C = z_idx
                elif block_idx == 3:
                    i_C = 1
                    k_C = 3 + z_idx
                
                W[k_C * n + i_C, col] += sign
            
            col += 1
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
