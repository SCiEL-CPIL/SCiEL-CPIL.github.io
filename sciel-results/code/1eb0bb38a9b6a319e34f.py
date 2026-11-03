"""
Program implementing Strassen-based rank minimization for 4x4x6 matrix multiplication.

Uses the optimal 49-term Strassen decomposition for the 4x4x4 part and
a 28-term decomposition for the 4x4x2 part, for a total rank of 77.
"""
import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # Strassen 2x2x2 terms
    # Each term has A_coeffs (2x2), B_coeffs (2x2), C_coeffs (2x2)
    # where the bilinear form is sum_{i,j} A_coeffs[i][j]*a[i][j] * sum_{i,j} B_coeffs[i][j]*b[i][j]
    # contributing to C via C_coeffs
    strassen_2x2_terms = [
        # M1 = (A00+A11)(B00+B11)
        {'A': [[1, 0], [0, 1]], 'B': [[1, 0], [0, 1]], 'C': [[1, 0], [0, 1]]},
        # M2 = (A10+A11)B00
        {'A': [[0, 0], [1, 1]], 'B': [[1, 0], [0, 0]], 'C': [[0, 0], [1, -1]]},
        # M3 = A00(B01-B11)
        {'A': [[1, 0], [0, 0]], 'B': [[0, 1], [0, -1]], 'C': [[0, 1], [0, 1]]},
        # M4 = A11(B10-B00)
        {'A': [[0, 0], [0, 1]], 'B': [[-1, 0], [1, 0]], 'C': [[1, 0], [1, 0]]},
        # M5 = (A00+A01)B11
        {'A': [[1, 1], [0, 0]], 'B': [[0, 0], [0, 1]], 'C': [[-1, 1], [0, 0]]},
        # M6 = (A10-A00)(B00+B01)
        {'A': [[-1, 0], [1, 0]], 'B': [[1, 1], [0, 0]], 'C': [[0, 0], [0, 1]]},
        # M7 = (A01-A11)(B10+B11)
        {'A': [[0, 1], [0, -1]], 'B': [[0, 0], [1, 1]], 'C': [[1, 0], [0, 0]]},
    ]
    
    # Build the full decomposition
    # We'll accumulate terms as (U_vec, V_vec, W_vec) tuples
    # U_vec has length n*m = 16
    # V_vec has length m*p = 24
    # W_vec has length p*n = 24
    
    terms = []
    
    def add_product_of_block_sums(a_blocks, b_blocks, c_blocks):
        """
        Add terms for the product of:
        X = sum of a_blocks[i] * A_block[i]
        Y = sum of b_blocks[i] * B_block[i]
        contributing to C = sum of c_blocks[i] * C_block[i]
        
        a_blocks: list of (row_offset, col_offset, coeff) for A blocks
        b_blocks: list of (row_offset, col_offset, coeff) for B blocks
        c_blocks: list of (row_offset, col_offset, coeff) for C blocks
        """
        for t in strassen_2x2_terms:
            u = [0] * (n * m)
            v = [0] * (m * p)
            w = [0] * (p * n)
            
            # Fill U
            for i in range(2):
                for j in range(2):
                    for (r, c, coeff) in a_blocks:
                        global_i = r + i
                        global_j = c + j
                        u[global_i * m + global_j] += coeff * t['A'][i][j]
            
            # Fill V
            for i in range(2):
                for j in range(2):
                    for (r, c, coeff) in b_blocks:
                        global_i = r + i
                        global_j = c + j
                        v[global_i * p + global_j] += coeff * t['B'][i][j]
            
            # Fill W
            for i in range(2):
                for j in range(2):
                    for (r, c, cc) in c_blocks:
                        global_row = r + i
                        global_col = c + j
                        w[global_col * n + global_row] += cc * t['C'][i][j]
            
            terms.append((u, v, w))
    
    # 4x4x4 part: C = A * B where A is 4x4, B is 4x4, C is 4x4
    # Using Strassen on 4x4x4:
    # Split into 2x2 blocks:
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    
    # M1 = (A00+A11)(B00+B11) contributes to C00 (+1) and C11 (+1)
    add_product_of_block_sums(
        [(0, 0, 1), (2, 2, 1)],  # A00 + A11
        [(0, 0, 1), (2, 2, 1)],  # B00 + B11
        [(0, 0, 1), (2, 2, 1)]   # C00 + C11
    )
    
    # M2 = (A10+A11)B00 contributes to C10 (+1) and C11 (-1)
    add_product_of_block_sums(
        [(2, 0, 1), (2, 2, 1)],  # A10 + A11
        [(0, 0, 1)],             # B00
        [(2, 0, 1), (2, 2, -1)]  # C10 - C11
    )
    
    # M3 = A00(B01-B11) contributes to C01 (+1) and C11 (+1)
    add_product_of_block_sums(
        [(0, 0, 1)],             # A00
        [(0, 2, 1), (2, 2, -1)], # B01 - B11
        [(0, 2, 1), (2, 2, 1)]   # C01 + C11
    )
    
    # M4 = A11(B10-B00) contributes to C00 (+1) and C10 (+1)
    add_product_of_block_sums(
        [(2, 2, 1)],             # A11
        [(2, 0, 1), (0, 0, -1)], # B10 - B00
        [(0, 0, 1), (2, 0, 1)]   # C00 + C10
    )
    
    # M5 = (A00+A01)B11 contributes to C00 (-1) and C01 (+1)
    add_product_of_block_sums(
        [(0, 0, 1), (0, 2, 1)],  # A00 + A01
        [(2, 2, 1)],             # B11
        [(0, 0, -1), (0, 2, 1)]  # -C00 + C01
    )
    
    # M6 = (A10-A00)(B00+B01) contributes to C11 (+1)
    add_product_of_block_sums(
        [(2, 0, 1), (0, 0, -1)], # A10 - A00
        [(0, 0, 1), (0, 2, 1)],  # B00 + B01
        [(2, 2, 1)]              # C11
    )
    
    # M7 = (A01-A11)(B10+B11) contributes to C00 (+1)
    add_product_of_block_sums(
        [(0, 2, 1), (2, 2, -1)], # A01 - A11
        [(2, 0, 1), (2, 2, 1)],  # B10 + B11
        [(0, 0, 1)]              # C00
    )
    
    # 4x4x2 part: C = A * B where A is 4x4, B is 4x2, C is 4x2
    # Split B into B_top (rows 0-1, cols 4-5) and B_bottom (rows 2-3, cols 4-5)
    # Note: B is 4x6, so cols 4-5 are the last 2 columns.
    # A is 4x4.
    # C is 4x6, so cols 4-5 are the last 2 columns.
    
    # C_top (rows 0-1 of C, cols 4-5) = A00 * B_top + A01 * B_bottom
    # A00 is A[0:2, 0:2], B_top is B[0:2, 4:6], C_top is C[0:2, 4:6]
    add_product_of_block_sums(
        [(0, 0, 1)],             # A00
        [(0, 4, 1)],             # B_top
        [(0, 4, 1)]              # C_top
    )
    add_product_of_block_sums(
        [(0, 2, 1)],             # A01
        [(2, 4, 1)],             # B_bottom
        [(0, 4, 1)]              # C_top
    )
    
    # C_bottom (rows 2-3 of C, cols 4-5) = A10 * B_top + A11 * B_bottom
    # A10 is A[2:4, 0:2], B_top is B[0:2, 4:6], C_bottom is C[2:4, 4:6]
    add_product_of_block_sums(
        [(2, 0, 1)],             # A10
        [(0, 4, 1)],             # B_top
        [(2, 4, 1)]              # C_bottom
    )
    add_product_of_block_sums(
        [(2, 2, 1)],             # A11
        [(2, 4, 1)],             # B_bottom
        [(2, 4, 1)]              # C_bottom
    )
    
    rank = len(terms)
    
    # Build U, V, W matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for k, (u, v, w) in enumerate(terms):
        for i in range(n * m):
            U[i][k] = u[i]
        for i in range(m * p):
            V[i][k] = v[i]
        for i in range(p * n):
            W[i][k] = w[i]
    
    return {"U": U, "V": V, "W": W}
