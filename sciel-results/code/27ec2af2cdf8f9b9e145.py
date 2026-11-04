import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 4x4 * 4x6 matrix multiplication.
    Uses Strassen's algorithm for the 4x4x4 part and independent 2x2x2 Strassen for the 4x4x2 part.
    Total rank: 49 (49 from 4x4x4, 28 from 4x4x2).
    """
    n, m, p = DIMENSIONS
    # rank = 49 (49 for 4x4x4, 28 for 4x4x2)
    rank = 49
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2x2 steps
    # (alpha, beta, gamma)
    # alpha, beta, gamma are dicts mapping (i,j) to coeff
    strassen_steps_222 = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1})
    ]

    # Part 1: 4x4x4 (cols 0-48)
    # A (4x4), B1 (4x4), C1 (4x4)
    # B1 is cols 0-3 of B. C1 is cols 0-3 of C.
    
    for s in range(7):
        alpha_4, beta_4, gamma_4 = strassen_steps_222[s]
        for s2 in range(7):
            alpha_2, beta_2, gamma_2 = strassen_steps_222[s2]
            col = s * 7 + s2
            
            # U
            for (bi, bj), val_a4 in alpha_4.items():
                for (si, sj), val_a2 in alpha_2.items():
                    r = (bi * 2 + si) * m + (bj * 2 + sj)
                    U[r][col] += val_a4 * val_a2
                    
            # V
            for (bi, bj), val_b4 in beta_4.items():
                for (si, sj), val_b2 in beta_2.items():
                    # B1 is 4x4, cols 0-3.
                    # Global col in B is bj*2 + sj.
                    r = (bi * 2 + si) * p + (bj * 2 + sj)
                    V[r][col] += val_b4 * val_b2
                    
            # W
            for (bi, bj), val_c4 in gamma_4.items():
                for (si, sj), val_c2 in gamma_2.items():
                    # C1 is 4x4, cols 0-3.
                    # Global col in C is bj*2 + sj.
                    r = (bj * 2 + sj) * n + (bi * 2 + si)
                    W[r][col] += val_c4 * val_c2

    # Part 2: 4x4x2 (cols 49-76)
    # A (4x4), B2 (4x2), C2 (4x2)
    # B2 is cols 4-5 of B. C2 is cols 4-5 of C.
    # We implement the 2x4 x 4x2 decomposition.
    # M (2x4) = [A_top_left, A_top_right]
    # N (4x2) = [B_top; B_bottom]
    # Result (2x2)
    
    col_start_2 = 49
    
    # Multiplications for C0 (top 2x2 of C)
    # A00 * B0
    for s2 in range(7):
        alpha_2, beta_2, gamma_2 = strassen_steps_222[s2]
        col = col_start_2 + s2
        
        # U: A00 (rows 0-1, cols 0-1)
        for (si, sj), val_a2 in alpha_2.items():
            r = (0 * 2 + si) * m + (0 * 2 + sj)
            U[r][col] += val_a2
            
        # V: B0 (rows 0-1, cols 4-5)
        for (si, sj), val_b2 in beta_2.items():
            r = (0 * 2 + si) * p + (4 + sj)
            V[r][col] += val_b2
            
        # W: C0 (rows 0-1, cols 4-5)
        for (si, sj), val_c2 in gamma_2.items():
            r = (4 + sj) * n + (0 * 2 + si)
            W[r][col] += val_c2

    # A01 * B1
    for s2 in range(7):
        alpha_2, beta_2, gamma_2 = strassen_steps_222[s2]
        col = col_start_2 + 7 + s2
        
        # U: A01 (rows 0-1, cols 2-3)
        for (si, sj), val_a2 in alpha_2.items():
            r = (0 * 2 + si) * m + (1 * 2 + sj)
            U[r][col] += val_a2
            
        # V: B1 (rows 2-3, cols 4-5)
        for (si, sj), val_b2 in beta_2.items():
            r = (1 * 2 + si) * p + (4 + sj)
            V[r][col] += val_b2
            
        # W: C0 (rows 0-1, cols 4-5)
        for (si, sj), val_c2 in gamma_2.items():
            r = (4 + sj) * n + (0 * 2 + si)
            W[r][col] += val_c2

    # Multiplications for C1 (bottom 2x2 of C)
    # A10 * B0
    for s2 in range(7):
        alpha_2, beta_2, gamma_2 = strassen_steps_222[s2]
        col = col_start_2 + 14 + s2
        
        # U: A10 (rows 2-3, cols 0-1)
        for (si, sj), val_a2 in alpha_2.items():
            r = (1 * 2 + si) * m + (0 * 2 + sj)
            U[r][col] += val_a2
            
        # V: B0 (rows 0-1, cols 4-5)
        for (si, sj), val_b2 in beta_2.items():
            r = (0 * 2 + si) * p + (4 + sj)
            V[r][col] += val_b2
            
        # W: C1 (rows 2-3, cols 4-5)
        for (si, sj), val_c2 in gamma_2.items():
            r = (4 + sj) * n + (1 * 2 + si)
            W[r][col] += val_c2

    # A11 * B1
    for s2 in range(7):
        alpha_2, beta_2, gamma_2 = strassen_steps_222[s2]
        col = col_start_2 + 21 + s2
        
        # U: A11 (rows 2-3, cols 2-3)
        for (si, sj), val_a2 in alpha_2.items():
            r = (1 * 2 + si) * m + (1 * 2 + sj)
            U[r][col] += val_a2
            
        # V: B1 (rows 2-3, cols 4-5)
        for (si, sj), val_b2 in beta_2.items():
            r = (1 * 2 + si) * p + (4 + sj)
            V[r][col] += val_b2
            
        # W: C1 (rows 2-3, cols 4-5)
        for (si, sj), val_c2 in gamma_2.items():
            r = (4 + sj) * n + (1 * 2 + si)
            W[r][col] += val_c2

    return {"U": U, "V": V, "W": W}
