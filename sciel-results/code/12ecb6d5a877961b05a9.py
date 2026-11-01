import time
import numpy as np

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal exact tensor decomposition for multiplying a 2x2 matrix by a 2x12 matrix.
    Uses Strassen's algorithm on 6 disjoint 2x2 blocks of the second matrix.
    Total rank = 6 * 7 = 42.
    
    This implementation uses a structured loop to build the factor matrices.
    """
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    num_pairs = p // 2
    rank = num_pairs * 7
    
    # Initialize U, V, W
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 multiplication
    # M1: (A00+A11)*(B00+B11)
    # M2: (A10+A11)*B00
    # M3: A00*(B01-B11)
    # M4: A11*(B10-B00)
    # M5: (A00+A01)*B11
    # M6: (A10-A00)*(B00+B01)
    # M7: (A01-A11)*(B10+B11)
    
    # Output combinations:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    for t in range(num_pairs):
        base = t * 7
        b_col_0 = 2 * t
        b_col_1 = 2 * t + 1
        
        for s in range(7):
            c = base + s
            
            if s == 0:
                U[0][c] = 1; U[3][c] = 1
                V[0*p + b_col_0][c] = 1; V[1*p + b_col_1][c] = 1
                W[b_col_0*n + 0][c] = 1; W[b_col_1*n + 1][c] = 1
            elif s == 1:
                U[2][c] = 1; U[3][c] = 1
                V[0*p + b_col_0][c] = 1
                W[b_col_0*n + 1][c] = 1; W[b_col_1*n + 1][c] = -1
            elif s == 2:
                U[0][c] = 1
                V[0*p + b_col_1][c] = 1; V[1*p + b_col_1][c] = -1
                W[b_col_1*n + 0][c] = 1; W[b_col_1*n + 1][c] = 1
            elif s == 3:
                U[3][c] = 1
                V[1*p + b_col_0][c] = 1; V[0*p + b_col_0][c] = -1
                W[b_col_0*n + 0][c] = 1; W[b_col_0*n + 1][c] = 1
            elif s == 4:
                U[0][c] = 1; U[1][c] = 1
                V[1*p + b_col_1][c] = 1
                W[b_col_0*n + 0][c] = -1; W[b_col_1*n + 0][c] = 1
            elif s == 5:
                U[2][c] = 1; U[0][c] = -1
                V[0*p + b_col_0][c] = 1; V[0*p + b_col_1][c] = 1
                W[b_col_1*n + 1][c] = 1
            elif s == 6:
                U[1][c] = 1; U[3][c] = -1
                V[1*p + b_col_0][c] = 1; V[1*p + b_col_1][c] = 1
                W[b_col_0*n + 0][c] = 1
                
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
