"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-42 exact tensor decomposition for multiplying a 2x2 matrix by a 2x12 matrix.
    Uses Strassen's algorithm for each 2x2x2 block (6 blocks total).
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=12
    
    num_blocks = p // 2  # 6 blocks of width 2
    
    rank = num_blocks * 7  # 42
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 24 x 42
    
    # Strassen factors for 2x2x2
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    for t in range(num_blocks):
        base = t * 7
        for s in range(7):
            c = base + s
            
            # B indices for this block
            # B00 is at V index 0*p + 2*t
            # B01 is at V index 0*p + 2*t + 1
            # B10 is at V index 1*p + 2*t
            # B11 is at V index 1*p + 2*t + 1
            v_b00 = 0 * p + 2 * t
            v_b01 = 0 * p + 2 * t + 1
            v_b10 = 1 * p + 2 * t
            v_b11 = 1 * p + 2 * t + 1
            
            # C indices for this block
            # C00 is at W index (2*t)*n + 0
            # C01 is at W index (2*t+1)*n + 0
            # C10 is at W index (2*t)*n + 1
            # C11 is at W index (2*t+1)*n + 1
            w_c00 = (2 * t) * n + 0
            w_c01 = (2 * t + 1) * n + 0
            w_c10 = (2 * t) * n + 1
            w_c11 = (2 * t + 1) * n + 1
            
            if s == 0: # M1
                U[0][c] = 1 # A00
                U[3][c] = 1 # A11
                V[v_b00][c] = 1 # B00
                V[v_b11][c] = 1 # B11
                W[w_c00][c] = 1 # C00 += M1
                W[w_c11][c] = 1 # C11 += M1
                
            elif s == 1: # M2
                U[2][c] = 1 # A10
                U[3][c] = 1 # A11
                V[v_b00][c] = 1 # B00
                W[w_c10][c] = 1 # C10 += M2
                W[w_c11][c] = -1 # C11 -= M2
                
            elif s == 2: # M3
                U[0][c] = 1 # A00
                V[v_b01][c] = 1 # B01
                V[v_b11][c] = -1 # -B11
                W[w_c01][c] = 1 # C01 += M3
                W[w_c11][c] = 1 # C11 += M3
                
            elif s == 3: # M4
                U[3][c] = 1 # A11
                V[v_b10][c] = 1 # B10
                V[v_b00][c] = -1 # -B00
                W[w_c00][c] = 1 # C00 += M4
                W[w_c10][c] = 1 # C10 += M4
                
            elif s == 4: # M5
                U[0][c] = 1 # A00
                U[1][c] = 1 # A01
                V[v_b11][c] = 1 # B11
                W[w_c00][c] = -1 # C00 -= M5
                W[w_c01][c] = 1 # C01 += M5
                
            elif s == 5: # M6
                U[2][c] = 1 # A10
                U[0][c] = -1 # -A00
                V[v_b00][c] = 1 # B00
                V[v_b01][c] = 1 # B01
                W[w_c11][c] = 1 # C11 += M6
                
            elif s == 6: # M7
                U[1][c] = 1 # A01
                U[3][c] = -1 # -A11
                V[v_b10][c] = 1 # B10
                V[v_b11][c] = 1 # B11
                W[w_c00][c] = 1 # C00 += M7
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
