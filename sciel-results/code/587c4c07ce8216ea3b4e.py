import numpy as np
import time

DIMENSIONS = [2, 2, 6]

def construct():
    """
    Returns a rank-21 decomposition for M_{2,2,6}.
    This uses Strassen's algorithm on three 2x2 blocks.
    """
    n, m, p = DIMENSIONS  # 2, 2, 6
    
    # Strassen terms for 2x2 matrix multiplication
    # U[i, r] coefficients for A
    # V[j, r] coefficients for B
    # W[k, r] coefficients for C
    
    # Strassen formula coefficients
    # A indices: 0:00, 1:01, 2:10, 3:11
    # B indices: 0:00, 1:01, 2:10, 3:11
    # C indices: 0:00, 1:01, 2:10, 3:11
    
    # The 7 Strassen products
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B00)
    # M4 = A11(B01 - B11)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A11 - A10)(B10 + B11)
    
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 - M6
    
    # Coefficients for each term r=0..6
    # U: A coeffs
    # V: B coeffs
    # W: C coeffs (dict of (i,k) -> coeff)
    
    strassen_terms = [
        # r=0: M1
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {(0, 0): 1, (1, 1): 1}),
        # r=1: M2
        ({2: 1, 3: 1}, {0: 1}, {(1, 0): 1, (1, 1): -1}),
        # r=2: M3
        ({0: 1}, {1: 1, 3: -1}, {(0, 1): 1, (1, 1): 1}),
        # r=3: M4
        ({3: 1}, {2: 1, 0: -1}, {(0, 0): 1, (1, 0): 1}),
        # r=4: M5
        ({0: 1, 1: 1}, {3: 1}, {(0, 0): -1, (0, 1): 1}),
        # r=5: M6
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {(1, 1): 1}),
        # r=6: M7
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {(0, 0): 1})
    ]
    
    # Initialize matrices
    # U: (n*m, rank) = (4, 21)
    # V: (m*p, rank) = (12, 21)
    # W: (p*n, rank) = (12, 21)
    # Note: W index is k*n + i. k is 0..5, i is 0..1.
    
    rank = 21
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(12)]
    W = [[0] * rank for _ in range(12)]
    
    r = 0
    for b in range(3):
        for t in range(7):
            u_coeffs, v_coeffs, w_coeffs = strassen_terms[t]
            
            # Fill U
            for idx, val in u_coeffs.items():
                U[idx][r] = val
                
            # Fill V
            for l, val in v_coeffs.items():
                # l is local B index 0..3
                # Global V index: row * p + col
                # l=0 (B00) -> row 0, col 2b
                # l=1 (B01) -> row 0, col 2b+1
                # l=2 (B10) -> row 1, col 2b
                # l=3 (B11) -> row 1, col 2b+1
                row = l // 2
                col = 2 * b + (l % 2)
                v_idx = row * p + col
                V[v_idx][r] = val
                
            # Fill W
            for (i, k_local), val in w_coeffs.items():
                # i is row 0..1
                # k_local is local col 0..1
                # Global W index: k_global * n + i
                # k_global = 2b + k_local
                k_global = 2 * b + k_local
                w_idx = k_global * n + i
                W[w_idx][r] = val
                
            r += 1
            
    return {"U": U, "V": V, "W": W}
