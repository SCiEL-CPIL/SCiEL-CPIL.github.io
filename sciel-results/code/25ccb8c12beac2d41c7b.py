import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # Rank 56 decomposition using Strassen on 2x2 blocks
    # A is 2x4, B is 4x8, C is 2x8
    # Split A into A1 (2x2, cols 0,1) and A2 (2x2, cols 2,3)
    # Split B into B1 (2x8, rows 0,1) and B2 (2x8, rows 2,3)
    # C = A1*B1 + A2*B2
    
    # We process 8 blocks of B (4 for B1, 4 for B2)
    # Each block is 2x2
    # Each block uses Strassen (7 mults)
    # Total rank = 8 * 7 = 56
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)] # 8x56
    V = [[0] * rank for _ in range(m * p)] # 32x56
    W = [[0] * rank for _ in range(p * n)] # 16x56
    
    # Strassen factors for 2x2 matrices
    # A indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # B indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    # M1: (A00+A11)(B00+B11)
    strassen_a = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0],# M6
        [0, 1, 0, -1] # M7
    ]
    strassen_b = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1],# M3
        [-1, 0, 1, 0],# M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    
    # Reconstruction coefficients for C
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    recon = {
        (0, 0): [(0, 1), (3, 1), (4, -1), (6, 1)],
        (0, 1): [(2, 1), (4, 1)],
        (1, 0): [(1, 1), (3, 1)],
        (1, 1): [(0, 1), (1, -1), (2, 1), (5, 1)]
    }
    
    col = 0
    # Process B1 blocks (rows 0,1 of B) with A1 (cols 0,1 of A)
    for j in range(4):
        # B block indices: rows 0,1; cols 2j, 2j+1
        b_row_0 = 0
        b_row_1 = 1
        b_col_0 = 2 * j
        b_col_1 = 2 * j + 1
        
        a_row_0 = 0
        a_row_1 = 1
        a_col_0 = 0
        a_col_1 = 1
        
        for s in range(7):
            # Fill U (A part)
            for a_idx in range(4):
                coeff = strassen_a[s][a_idx]
                if coeff != 0:
                    # Map a_idx to A index
                    # A is 2x4. A1 is cols 0,1.
                    # a_idx 0 -> (0,0), 1 -> (0,1), 2 -> (1,0), 3 -> (1,1)
                    r = (a_idx // 2)
                    c = (a_idx % 2)
                    # Global index in A (2x4)
                    global_a_idx = r * 4 + (a_col_0 + c)
                    U[global_a_idx][col] += coeff
            
            # Fill V (B part)
            for b_idx in range(4):
                coeff = strassen_b[s][b_idx]
                if coeff != 0:
                    # Map b_idx to B index
                    # B is 4x8. B1 is rows 0,1.
                    # b_idx 0 -> (0,0), 1 -> (0,1), 2 -> (1,0), 3 -> (1,1)
                    r = (b_idx // 2)
                    c = (b_idx % 2)
                    # Global index in B (4x8)
                    global_b_idx = (b_row_0 + r) * 8 + (b_col_0 + c)
                    V[global_b_idx][col] += coeff
            
            # Fill W (C part)
            # The result of this 2x2 product contributes to C[0:2, 2j:2j+2]
            # C is 2x8.
            for (c_r, c_c), terms in recon.items():
                for (m_idx, coeff) in terms:
                    if m_idx == s:
                        global_c_idx = c_r * 8 + (b_col_0 + c_c)
                        W[global_c_idx][col] += coeff
                        
            col += 1
            
    # Process B2 blocks (rows 2,3 of B) with A2 (cols 2,3 of A)
    for j in range(4):
        # B block indices: rows 2,3; cols 2j, 2j+1
        b_row_0 = 2
        b_row_1 = 3
        b_col_0 = 2 * j
        b_col_1 = 2 * j + 1
        
        a_row_0 = 0
        a_row_1 = 1
        a_col_0 = 2
        a_col_1 = 3
        
        for s in range(7):
            # Fill U (A part)
            for a_idx in range(4):
                coeff = strassen_a[s][a_idx]
                if coeff != 0:
                    r = (a_idx // 2)
                    c = (a_idx % 2)
                    global_a_idx = r * 4 + (a_col_0 + c)
                    U[global_a_idx][col] += coeff
            
            # Fill V (B part)
            for b_idx in range(4):
                coeff = strassen_b[s][b_idx]
                if coeff != 0:
                    r = (b_idx // 2)
                    c = (b_idx % 2)
                    global_b_idx = (b_row_0 + r) * 8 + (b_col_0 + c)
                    V[global_b_idx][col] += coeff
            
            # Fill W (C part)
            for (c_r, c_c), terms in recon.items():
                for (m_idx, coeff) in terms:
                    if m_idx == s:
                        global_c_idx = c_r * 8 + (b_col_0 + c_c)
                        W[global_c_idx][col] += coeff
                        
            col += 1
            
    return {"U": U, "V": V, "W": W}
