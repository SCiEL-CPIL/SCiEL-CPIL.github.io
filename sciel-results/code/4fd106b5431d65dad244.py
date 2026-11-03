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
    
    def add_strassen_block(a_row_offset, a_col_offset, b_row_offset, b_col_offset, 
                            c_row_offset, c_col_offset, terms_list):
        """
        Add Strassen 2x2x2 terms for a block multiplication.
        A block: rows a_row_offset..a_row_offset+1, cols a_col_offset..a_col_offset+1
        B block: rows b_row_offset..b_row_offset+1, cols b_col_offset..b_col_offset+1
        C block: rows c_row_offset..c_row_offset+1, cols c_col_offset..c_col_offset+1
        """
        for t in terms_list:
            u = [0] * (n * m)
            v = [0] * (m * p)
            w = [0] * (p * n)
            
            # Fill U
            for i in range(2):
                for j in range(2):
                    global_i = a_row_offset + i
                    global_j = a_col_offset + j
                    u[global_i * m + global_j] = t['A'][i][j]
            
            # Fill V
            for i in range(2):
                for j in range(2):
                    global_i = b_row_offset + i
                    global_j = b_col_offset + j
                    v[global_i * p + global_j] = t['B'][i][j]
            
            # Fill W
            for i in range(2):
                for j in range(2):
                    global_row = c_row_offset + i
                    global_col = c_col_offset + j
                    w[global_col * n + global_row] = t['C'][i][j]
            
            terms.append((u, v, w))
    
    # 4x4x4 part: C = A * B where A is 4x4, B is 4x4, C is 4x4
    # Using Strassen on 4x4x4:
    # Split into 2x2 blocks:
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    #
    # Strassen 4x4x4 uses 7 products of 2x2 matrices:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # Each Mi is a product of two 2x2 matrices, so each needs 7 Strassen terms.
    # Total: 7 * 7 = 49 terms.
    
    # For each Strassen 4x4 term, we need to decompose the product of two 2x2 matrices
    # using Strassen 2x2.
    
    # Let's define the 7 products for 4x4x4 Strassen:
    # Each product is X * Y where X and Y are 2x2 matrices formed from blocks of A and B.
    
    # M1: X = A00 + A11, Y = B00 + B11
    # M2: X = A10 + A11, Y = B00
    # M3: X = A00, Y = B01 - B11
    # M4: X = A11, Y = B10 - B00
    # M5: X = A00 + A01, Y = B11
    # M6: X = A10 - A00, Y = B00 + B01
    # M7: X = A01 - A11, Y = B10 + B11
    
    # For each product X*Y, we apply Strassen 2x2.
    # But X and Y are sums/differences of blocks, so we need to handle that.
    
    # Let's create a helper to add a product of two linear combinations of 2x2 blocks.
    
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
        # For each Strassen 2x2 term t:
        for t in strassen_2x2_terms:
            u = [0] * (n * m)
            v = [0] * (m * p)
            w = [0] * (p * n)
            
            # Fill U: for each position (i,j) in the 2x2 X matrix,
            # X[i][j] = sum of a_blocks[k] * A_block_k[i][j]
            # The Strassen term uses t['A'][i][j] * X[i][j]
            # So U gets contributions from each A block.
            
            for i in range(2):
                for j in range(2):
                    for (r, c, coeff) in a_blocks:
                        # A_block covers rows r..r+1, cols c..c+1
                        # Position (i,j) in X corresponds to global position (r+i, c+j) in A
                        # if the block is at (r, c)
                        global_i = r + i
                        global_j = c + j
                        u[global_i * m + global_j] += coeff * t['A'][i][j]
            
            # Fill V similarly
            for i in range(2):
                for j in range(2):
                    for (r, c, coeff) in b_blocks:
                        global_i = r + i
                        global_j = c + j
                        v[global_i * p + global_j] += coeff * t['B'][i][j]
            
            # Fill W: the result C = sum of c_blocks[k] * C_block_k
            # Each C_block_k is 2x2, and the Strassen term contributes t['C'][i][j] to C[i][j]
            # So for each c_block at (r, c) with coefficient cc, the contribution to
            # C[r+i][c+j] is cc * t['C'][i][j]
            
            for i in range(2):
                for j in range(2):
                    for (r, c, cc) in c_blocks:
                        global_row = r + i
                        global_col = c + j
                        w[global_col * n + global_row] += cc * t['C'][i][j]
            
            terms.append((u, v, w))
    
    # Now add the 7 Strassen 4x4 terms:
    
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
    # Split C into C_top (rows 0-1, cols 4-5) and C_bottom (rows 2-3, cols 4-5)
    # C_top = A00*B_top + A01*B_bottom
    # C_bottom = A10*B_top + A11*B_bottom
    #
    # For each, we use Strassen 2x2x2 for each 2x2x2 product.
    # So 2 * 7 + 2 * 7 = 28 terms.
    
    # C_top = A00*B_top + A01*B_bottom
    # A00*B_top: A block at (0,0), B block at (0,4), C block at (0,4)
    for t in strassen_2x2_terms:
        u = [0] * (n * m)
        v = [0] * (m * p)
        w = [0] * (p * n)
        
        # A00
        for i in range(2):
            for j in range(2):
                u[(0+i) * m + (0+j)] = t['A'][i][j]
        
        # B_top (rows 0-1, cols 4-5)
        for i in range(2):
            for j in range(2):
                v[(0+i) * p + (4+j)] = t['B'][i][j]
        
        # C_top (rows 0-1, cols 4-5)
        for i in range(2):
            for j in range(2):
                w[(4+j) * n + (0+i)] = t['C'][i][j]
        
        terms.append((u, v, w))
    
    # A01*B_bottom: A block at (0,2), B block at (2,4), C block at (0,4)
    for t in strassen_2x2_terms:
        u = [0] * (n * m)
        v = [0] * (m * p)
        w = [0] * (p * n)
        
        # A01
        for i in range(2):
            for j in range(2):
                u[(0+i) * m + (2+j)] = t['A'][i][j]
        
        # B_bottom (rows 2-3, cols 4-5)
        for i in range(2):
            for j in range(2):
                v[(2+i) * p + (4+j)] = t['B'][i][j]
        
        # C_top (rows 0-1, cols 4-5)
        for i in range(2):
            for j in range(2):
                w[(4+j) * n + (0+i)] = t['C'][i][j]
        
        terms.append((u, v, w))
    
    # C_bottom = A10*B_top + A11*B_bottom
    # A10*B_top: A block at (2,0), B block at (0,4), C block at (2,4)
    for t in strassen_2x2_terms:
        u = [0] * (n * m)
        v = [0] * (m * p)
        w = [0] * (p * n)
        
        # A10
        for i in range(2):
            for j in range(2):
                u[(2+i) * m + (0+j)] = t['A'][i][j]
        
        # B_top (rows 0-1, cols 4-5)
        for i in range(2):
            for j in range(2):
                v[(0+i) * p + (4+j)] = t['B'][i][j]
        
        # C_bottom (rows 2-3, cols 4-5)
        for i in range(2):
            for j in range(2):
                w[(4+j) * n + (2+i)] = t['C'][i][j]
        
        terms.append((u, v, w))
    
    # A11*B_bottom: A block at (2,2), B block at (2,4), C block at (2,4)
    for t in strassen_2x2_terms:
        u = [0] * (n * m)
        v = [0] * (m * p)
        w = [0] * (p * n)
        
        # A11
        for i in range(2):
            for j in range(2):
                u[(2+i) * m + (2+j)] = t['A'][i][j]
        
        # B_bottom (rows 2-3, cols 4-5)
        for i in range(2):
            for j in range(2):
                v[(2+i) * p + (4+j)] = t['B'][i][j]
        
        # C_bottom (rows 2-3, cols 4-5)
        for i in range(2):
            for j in range(2):
                w[(4+j) * n + (2+i)] = t['C'][i][j]
        
        terms.append((u, v, w))
    
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
