import numpy as np
import time

def construct():
    n, m, p = 2, 4, 4
    rank = 28
    
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    
    # Helper to add a Strassen block
    # a_row, a_col: indices of A submatrix (2x2)
    # b_row, b_col: indices of B submatrix (2x2)
    # c_row, c_col: indices of C submatrix (2x2)
    # offset: starting rank index
    def add_strassen_block(a_r0, a_c0, b_r0, b_c0, c_r0, c_c0, offset):
        # Local indices 0,1 for rows/cols
        # Global A index: (a_r0 + i) * m + (a_c0 + j)
        # Global B index: (b_r0 + j) * p + (b_c0 + k)
        # Global C index: (c_r0 + i) * p + (c_c0 + k) -> W index is (c_c0+k)*n + (c_r0+i)
        
        # Strassen M1..M7 definitions
        # M1: A11 * B11
        # M2: (A10 - A11) * (B11 + B10)
        # M3: (A00 + A11) * (B00 + B11)
        # M4: (A01 - A00) * (B00 + B10)
        # M5: (A10 + A00) * (B01 - B00)
        # M6: (A11 - A01) * (B01 + B11)
        # M7: (A00 - A10) * (B10 - B11)
        
        # Outputs
        # C00 = M3 + M7 - M5 + M2
        # C01 = M1 + M5
        # C10 = M2 + M4
        # C11 = M3 - M1 + M4 + M6
        
        # Define the 7 terms
        # Each term is a tuple of (A_coeffs, B_coeffs, C_coeffs)
        # A_coeffs: dict {(i,j): val}
        # B_coeffs: dict {(j,k): val}
        # C_coeffs: dict {(i,k): val}
        
        terms = [
            # M1
            (
                {(1,1): 1}, 
                {(1,1): 1}, 
                {(0,1): 1, (1,0): 0, (1,1): -1, (0,0): 0} # Wait, C coeffs are for the final C
            ),
            # M2
            (
                {(1,0): 1, (1,1): -1}, 
                {(1,1): 1, (1,0): 1}, 
                {(0,0): 1, (1,0): 1, (0,1): 0, (1,1): 0}
            ),
            # M3
            (
                {(0,0): 1, (1,1): 1}, 
                {(0,0): 1, (1,1): 1}, 
                {(0,0): 1, (1,1): 1, (0,1): 0, (1,0): 0}
            ),
            # M4
            (
                {(0,1): 1, (0,0): -1}, 
                {(0,0): 1, (1,0): 1}, 
                {(1,0): 1, (1,1): 1, (0,0): 0, (0,1): 0}
            ),
            # M5
            (
                {(1,0): 1, (0,0): 1}, 
                {(0,1): 1, (0,0): -1}, 
                {(0,1): 1, (0,0): -1, (1,0): 0, (1,1): 0}
            ),
            # M6
            (
                {(1,1): 1, (0,1): -1}, 
                {(0,1): 1, (1,1): 1}, 
                {(1,1): 1, (0,0): 0, (0,1): 0, (1,0): 0}
            ),
            # M7
            (
                {(0,0): 1, (1,0): -1}, 
                {(1,0): 1, (1,1): -1}, 
                {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 0}
            ),
        ]
        
        # Note: The C_coeffs in the tuples above are the contributions to the LOCAL C matrix.
        # We need to map local (i,k) to global C index.
        
        for t_idx, (a_coeff, b_coeff, c_coeff) in enumerate(terms):
            r = offset + t_idx
            
            # Fill U
            for (i, j), val in a_coeff.items():
                gi = a_r0 + i
                gj = a_c0 + j
                U[gi * m + gj, r] = val
                
            # Fill V
            for (j, k), val in b_coeff.items():
                gj = b_r0 + j
                gk = b_c0 + k
                V[gj * p + gk, r] = val
                
            # Fill W
            # W index is (k*n + i)
            for (i, k), val in c_coeff.items():
                gi = c_r0 + i
                gk = c_c0 + k
                W[gk * n + gi, r] = val

    # Block 1: A_L (cols 0-1), B_TL (rows 0-1, cols 0-1) -> C_L (rows 0-1, cols 0-1)
    # A is 2x4. A_L is A[:, 0:2]. So a_r0=0, a_c0=0.
    # B is 4x4. B_TL is B[0:2, 0:2]. So b_r0=0, b_c0=0.
    # C is 2x4. C_L is C[:, 0:2]. So c_r0=0, c_c0=0.
    add_strassen_block(0, 0, 0, 0, 0, 0, 0)
    
    # Block 2: A_L (cols 0-1), B_TR (rows 0-1, cols 2-3) -> C_R (rows 0-1, cols 2-3)
    # A_L is A[:, 0:2]. a_r0=0, a_c0=0.
    # B_TR is B[0:2, 2:4]. b_r0=0, b_c0=2.
    # C_R is C[:, 2:4]. c_r0=0, c_c0=2.
    add_strassen_block(0, 0, 0, 2, 0, 2, 7)
    
    # Block 3: A_R (cols 2-3), B_BL (rows 2-3, cols 0-1) -> C_L (rows 0-1, cols 0-1)
    # A_R is A[:, 2:4]. a_r0=0, a_c0=2.
    # B_BL is B[2:4, 0:2]. b_r0=2, b_c0=0.
    # C_L is C[:, 0:2]. c_r0=0, c_c0=0.
    add_strassen_block(0, 2, 2, 0, 0, 0, 14)
    
    # Block 4: A_R (cols 2-3), B_BR (rows 2-3, cols 2-3) -> C_R (rows 0-1, cols 2-3)
    # A_R is A[:, 2:4]. a_r0=0, a_c0=2.
    # B_BR is B[2:4, 2:4]. b_r0=2, b_c0=2.
    # C_R is C[:, 2:4]. c_r0=0, c_c0=2.
    add_strassen_block(0, 2, 2, 2, 0, 2, 21)
    
    return {'U': U, 'V': V, 'W': W}
