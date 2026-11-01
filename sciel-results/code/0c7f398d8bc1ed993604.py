import numpy as np
import time

DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 56 decomposition using Strassen for 2x2 blocks
    # 8 blocks of 2 columns each
    num_blocks = p // 2
    rank = num_blocks * 7
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for A (i, j) -> index i*m+j
    # A = [[a, b], [c, d]]
    # Indices: 0: a, 1: b, 2: c, 3: d
    # M1: a+d
    # M2: c+d
    # M3: a
    # M4: d
    # M5: a+b
    # M6: c-a
    # M7: b-d
    
    u_coeffs = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0], # M6
        [0, 1, 0, -1]  # M7
    ]
    
    # Strassen coefficients for B block (e, f, g, h)
    # B_block = [[e, f], [g, h]]
    # e = B[0, 2b], f = B[0, 2b+1]
    # g = B[1, 2b], h = B[1, 2b+1]
    # V indices:
    # e -> 0*p + 2b = 2b
    # f -> 0*p + 2b+1 = 2b+1
    # g -> 1*p + 2b = 16+2b
    # h -> 1*p + 2b+1 = 16+2b+1
    
    # M1: e+h
    # M2: e
    # M3: f-h
    # M4: g-e
    # M5: h
    # M6: e+f
    # M7: g+h
    
    # Strassen coefficients for C block
    # C_block = [[C00, C01], [C10, C11]]
    # W indices:
    # C00 -> 2b*n + 0 = 4b
    # C01 -> 2b*n + 1 = 4b+1
    # C10 -> (2b+1)*n + 0 = 4b+2
    # C11 -> (2b+1)*n + 1 = 4b+3
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    w_coeffs = [
        [1, 0, 0, 1], # M1: C00=1, C11=1
        [0, 0, 1, -1], # M2: C10=1, C11=-1
        [0, 1, 0, 1], # M3: C01=1, C11=1
        [1, 0, 1, 0], # M4: C00=1, C10=1
        [-1, 1, 0, 0], # M5: C00=-1, C01=1
        [0, 0, 0, 1], # M6: C11=1
        [1, 0, 0, 0]  # M7: C00=1
    ]
    
    for b in range(num_blocks):
        col_start = b * 7
        # V indices for this block
        idx_e = 2 * b
        idx_f = 2 * b + 1
        idx_g = 16 + 2 * b
        idx_h = 16 + 2 * b + 1
        
        # W indices for this block
        idx_c00 = 4 * b
        idx_c01 = 4 * b + 1
        idx_c10 = 4 * b + 2
        idx_c11 = 4 * b + 3
        
        for s in range(7):
            r = col_start + s
            
            # Fill U
            for i in range(4):
                U[i][r] = u_coeffs[s][i]
                
            # Fill V
            # M1: e+h
            # M2: e
            # M3: f-h
            # M4: g-e
            # M5: h
            # M6: e+f
            # M7: g+h
            
            if s == 0: # e+h
                V[idx_e][r] = 1
                V[idx_h][r] = 1
            elif s == 1: # e
                V[idx_e][r] = 1
            elif s == 2: # f-h
                V[idx_f][r] = 1
                V[idx_h][r] = -1
            elif s == 3: # g-e
                V[idx_g][r] = 1
                V[idx_e][r] = -1
            elif s == 4: # h
                V[idx_h][r] = 1
            elif s == 5: # e+f
                V[idx_e][r] = 1
                V[idx_f][r] = 1
            elif s == 6: # g+h
                V[idx_g][r] = 1
                V[idx_h][r] = 1
            
            # Fill W
            # w_coeffs[s] is [C00, C01, C10, C11]
            W[idx_c00][r] = w_coeffs[s][0]
            W[idx_c01][r] = w_coeffs[s][1]
            W[idx_c10][r] = w_coeffs[s][2]
            W[idx_c11][r] = w_coeffs[s][3]
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
