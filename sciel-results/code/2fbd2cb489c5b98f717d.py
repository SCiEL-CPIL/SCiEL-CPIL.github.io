import time
import numpy as np

DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 8
    
    # Strassen's 7-term algorithm for 2x2 matrix multiplication.
    # We group the 8 columns of B into 4 pairs.
    # Each pair is a 2x2 sub-matrix.
    # We apply Strassen to each pair independently.
    # Total rank: 4 * 7 = 28.
    
    rank = 28
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen products definitions
    # A_coeffs: {(i,j): coeff}
    # B_coeffs: {(j,k_sub): coeff}
    # C_contrib: {(i, k_sub): coeff}
    strassen_products = [
        # M1 = (A00+A11)(B00+B11) -> C00+=M1, C11+=M1
        {'A': {(0,0): 1, (1,1): 1},
         'B': {(0,0): 1, (1,1): 1},
         'C': {(0,0): 1, (1,1): 1}},
        # M2 = (A10+A11)B00 -> C10+=M2, C11-=M2
        {'A': {(1,0): 1, (1,1): 1},
         'B': {(0,0): 1},
         'C': {(1,0): 1, (1,1): -1}},
        # M3 = A00(B01-B11) -> C01+=M3, C11+=M3
        {'A': {(0,0): 1},
         'B': {(0,1): 1, (1,1): -1},
         'C': {(0,1): 1, (1,1): 1}},
        # M4 = A11(B10-B00) -> C00+=M4, C10+=M4
        {'A': {(1,1): 1},
         'B': {(1,0): 1, (0,0): -1},
         'C': {(0,0): 1, (1,0): 1}},
        # M5 = (A00+A01)B11 -> C00-=M5, C01+=M5
        {'A': {(0,0): 1, (0,1): 1},
         'B': {(1,1): 1},
         'C': {(0,0): -1, (0,1): 1}},
        # M6 = (A10-A00)(B00+B01) -> C11+=M6
        {'A': {(1,0): 1, (0,0): -1},
         'B': {(0,0): 1, (0,1): 1},
         'C': {(1,1): 1}},
        # M7 = (A01-A11)(B10+B11) -> C00+=M7
        {'A': {(0,1): 1, (1,1): -1},
         'B': {(1,0): 1, (1,1): 1},
         'C': {(0,0): 1}},
    ]
    
    for q in range(p // 2):
        k1 = 2 * q
        k2 = 2 * q + 1
        for r, prod in enumerate(strassen_products):
            t = q * 7 + r
            
            for (i, j), coeff in prod['A'].items():
                U[i * m + j, t] = coeff
            
            for (j, k_sub), coeff in prod['B'].items():
                k_global = k1 + k_sub
                V[j * p + k_global, t] = coeff
            
            for (i, k_sub), coeff in prod['C'].items():
                k_global = k1 + k_sub
                W[k_global * n + i, t] = coeff
    
    # Verification is skipped for speed, as the construction is analytically correct.
    # If verification is required, it can be added, but it takes ~0.1s.
    # Given the 120s limit, we are safe.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
