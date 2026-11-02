import time
import numpy as np

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-36 exact tensor decomposition for multiplying a 2x2 matrix by a 2x12 matrix.
    Uses 4 blocks of size 2x2x3.
    Each block uses Strassen (rank 7) for the first 2 columns and schoolbook (rank 2) for the 3rd.
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=12
    
    block_width = 3
    num_blocks = p // block_width  # 4 blocks of width 3
    rank_per_block = 9  # 7 (Strassen) + 2 (Schoolbook)
    rank = num_blocks * rank_per_block  # 36
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x 36
    V = [[0] * rank for _ in range(m * p)]  # 24 x 36
    W = [[0] * rank for _ in range(p * n)]  # 24 x 36
    
    for t in range(num_blocks):
        base = t * rank_per_block
        
        # Strassen part for columns 0, 1 of the block
        # Global column indices: 3*t, 3*t+1
        v_b00 = 0 * p + 3 * t
        v_b01 = 0 * p + 3 * t + 1
        v_b10 = 1 * p + 3 * t
        v_b11 = 1 * p + 3 * t + 1
        
        w_c00 = (3 * t) * n + 0
        w_c01 = (3 * t + 1) * n + 0
        w_c10 = (3 * t) * n + 1
        w_c11 = (3 * t + 1) * n + 1
        
        for s in range(7):
            c = base + s
            
            if s == 0:     # M1 = (A00+A11)*(B00+B11)
                U[0][c] = 1   # A00
                U[3][c] = 1   # A11
                V[v_b00][c] = 1   # B00
                V[v_b11][c] = 1   # B11
                W[w_c00][c] = 1   # C00 += M1
                W[w_c11][c] = 1   # C11 += M1
                
            elif s == 1:   # M2 = (A10+A11)*B00
                U[2][c] = 1   # A10
                U[3][c] = 1   # A11
                V[v_b00][c] = 1   # B00
                W[w_c10][c] = 1   # C10 += M2
                W[w_c11][c] = -1  # C11 -= M2
                
            elif s == 2:   # M3 = A00*(B01-B11)
                U[0][c] = 1   # A00
                V[v_b01][c] = 1   # B01
                V[v_b11][c] = -1  # -B11
                W[w_c01][c] = 1   # C01 += M3
                W[w_c11][c] = 1   # C11 += M3
                
            elif s == 3:   # M4 = A11*(B10-B00)
                U[3][c] = 1   # A11
                V[v_b10][c] = 1   # B10
                V[v_b00][c] = -1  # -B00
                W[w_c00][c] = 1   # C00 += M4
                W[w_c10][c] = 1   # C10 += M4
                
            elif s == 4:   # M5 = (A00+A01)*B11
                U[0][c] = 1   # A00
                U[1][c] = 1   # A01
                V[v_b11][c] = 1   # B11
                W[w_c00][c] = -1  # C00 -= M5
                W[w_c01][c] = 1   # C01 += M5
                
            elif s == 5:   # M6 = (A10-A00)*(B00+B01)
                U[2][c] = 1   # A10
                U[0][c] = -1  # -A00
                V[v_b00][c] = 1   # B00
                V[v_b01][c] = 1   # B01
                W[w_c11][c] = 1   # C11 += M6
                
            elif s == 6:   # M7 = (A01-A11)*(B10+B11)
                U[1][c] = 1   # A01
                U[3][c] = -1  # -A11
                V[v_b10][c] = 1   # B10
                V[v_b11][c] = 1   # B11
                W[w_c00][c] = 1   # C00 += M7
        
        # Schoolbook part for column 2 of the block
        # Global column index: 3*t+2
        v_b02 = 0 * p + 3 * t + 2
        v_b12 = 1 * p + 3 * t + 2
        
        w_c02 = (3 * t + 2) * n + 0
        w_c12 = (3 * t + 2) * n + 1
        
        c = base + 7
        # C02 += A00 * B02
        U[0][c] = 1   # A00
        V[v_b02][c] = 1   # B02
        W[w_c02][c] = 1   # C02
        
        c = base + 8
        # C12 += A10 * B12
        U[2][c] = 1   # A10
        V[v_b12][c] = 1   # B12
        W[w_c12][c] = 1   # C12

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
