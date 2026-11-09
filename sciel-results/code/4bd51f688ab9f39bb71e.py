import numpy as np

DIMENSIONS = [4, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # Rank 98
    rank = 98
    
    # Strassen 2x2 coefficients
    # u, v, w are lists of 7 tuples of 4 coeffs
    # Order: 00, 01, 10, 11
    S_u = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0],# M6
        [0, 1, -1, 0] # M7
    ]
    S_v = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1],# M3
        [-1, 0, 1, 0],# M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    S_w = [
        [1, 0, 0, 1], # M1 -> C00, C11
        [0, 0, 1, -1],# M2 -> C10, C11
        [0, 1, 0, 1], # M3 -> C01, C11
        [1, 0, 1, 0], # M4 -> C00, C10
        [-1, 1, 0, 0],# M5 -> C00, C01
        [0, 0, 0, 1], # M6 -> C11
        [1, 0, 0, 0]  # M7 -> C00
    ]

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Generate 49 terms for Left half (B_L, C_L)
    for s in range(7):
        for t in range(7):
            # Calculate global coefficients
            # A is 4x4. Block (bi, bj) element (ei, ej)
            # Global A index: (2*bi + ei)*4 + (2*bj + ej)
            # S_u[s] gives coeffs for blocks. S_u[t] gives coeffs for elements within block.
            # Wait, S_u is for the outer Strassen. S_u[t] is for the inner Strassen.
            # The outer Strassen combines A blocks.
            # The inner Strassen combines elements of the resulting 2x2 matrix.
            
            # U coefficients
            for bi in range(2):
                for bj in range(2):
                    coeff_block = S_u[s][bi*2+bj] # Wait, S_u index is 00,01,10,11 -> 0,1,2,3
                    # S_u[s] is [u00, u01, u10, u11]
                    # My S_u list above:
                    # S_u[0] = [1, 0, 0, 1] -> u00=1, u01=0, u10=0, u11=1. Correct.
                    
                    if coeff_block == 0: continue
                    
                    for ei in range(2):
                        for ej in range(2):
                            coeff_elem = S_u[t][ei*2+ej]
                            if coeff_elem == 0: continue
                            
                            global_i = 2*bi + ei
                            global_j = 2*bj + ej
                            idx_a = global_i * m + global_j
                            U[idx_a][col] += coeff_block * coeff_elem
            
            # V coefficients (for B_L)
            # B_L is 4x4.
            for bi in range(2):
                for bj in range(2):
                    coeff_block = S_v[s][bi*2+bj]
                    if coeff_block == 0: continue
                    for ei in range(2):
                        for ej in range(2):
                            coeff_elem = S_v[t][ei*2+ej]
                            if coeff_elem == 0: continue
                            
                            global_i = 2*bi + ei
                            global_j = 2*bj + ej
                            # B_L is cols 0-3 of B (which is 4x8)
                            # B index: global_i * p + global_j
                            idx_b = global_i * p + global_j
                            V[idx_b][col] += coeff_block * coeff_elem
            
            # W coefficients (for C_L)
            # C_L is 4x4 (cols 0-3 of C, which is 8x4? No, C is 4x8)
            # C is n x p = 4 x 8.
            # W index: k*n + i. k is col (0-7), i is row (0-3).
            # C_L is cols 0-3.
            for bi in range(2):
                for bj in range(2):
                    coeff_block = S_w[s][bi*2+bj]
                    if coeff_block == 0: continue
                    for ei in range(2):
                        for ej in range(2):
                            coeff_elem = S_w[t][ei*2+ej]
                            if coeff_elem == 0: continue
                            
                            global_i = 2*bi + ei # Row of C
                            global_j = 2*bj + ej # Col of C (within L)
                            # W index: k*n + i. k=global_j, i=global_i
                            idx_c = global_j * n + global_i
                            W[idx_c][col] += coeff_block * coeff_elem
            
            col += 1
            
    # Generate 49 terms for Right half (B_R, C_R)
    for s in range(7):
        for t in range(7):
            # U is same as above
            for bi in range(2):
                for bj in range(2):
                    coeff_block = S_u[s][bi*2+bj]
                    if coeff_block == 0: continue
                    for ei in range(2):
                        for ej in range(2):
                            coeff_elem = S_u[t][ei*2+ej]
                            if coeff_elem == 0: continue
                            
                            global_i = 2*bi + ei
                            global_j = 2*bj + ej
                            idx_a = global_i * m + global_j
                            U[idx_a][col] += coeff_block * coeff_elem
            
            # V coefficients (for B_R)
            # B_R is cols 4-7 of B.
            for bi in range(2):
                for bj in range(2):
                    coeff_block = S_v[s][bi*2+bj]
                    if coeff_block == 0: continue
                    for ei in range(2):
                        for ej in range(2):
                            coeff_elem = S_v[t][ei*2+ej]
                            if coeff_elem == 0: continue
                            
                            global_i = 2*bi + ei
                            global_j = 2*bj + ej
                            # B_R col offset is 4
                            idx_b = global_i * p + (global_j + 4)
                            V[idx_b][col] += coeff_block * coeff_elem
            
            # W coefficients (for C_R)
            # C_R is cols 4-7 of C.
            for bi in range(2):
                for bj in range(2):
                    coeff_block = S_w[s][bi*2+bj]
                    if coeff_block == 0: continue
                    for ei in range(2):
                        for ej in range(2):
                            coeff_elem = S_w[t][ei*2+ej]
                            if coeff_elem == 0: continue
                            
                            global_i = 2*bi + ei
                            global_j = 2*bj + ej
                            # C_R col offset is 4
                            idx_c = (global_j + 4) * n + global_i
                            W[idx_c][col] += coeff_block * coeff_elem
            
            col += 1

    return {"U": U, "V": V, "W": W}
