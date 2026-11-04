import numpy as np
import time

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Initialize U, V, W
    # We will build the solution term by term.
    # Each term is (u, v, w)
    terms = []
    
    # Helper to add a term
    def add_term(u_indices, u_coeffs, v_indices, v_coeffs, w_indices, w_coeffs):
        u = np.zeros(nm)
        for idx, coeff in zip(u_indices, u_coeffs):
            u[idx] += coeff
        v = np.zeros(mp)
        for idx, coeff in zip(v_indices, v_coeffs):
            v[idx] += coeff
        w = np.zeros(pn)
        for idx, coeff in zip(w_indices, w_coeffs):
            w[idx] += coeff
        terms.append((u, v, w))

    # Indices for A (2x5): i*5 + j
    # Indices for B (5x6): j*6 + k
    # Indices for C (6x2): k*2 + i
    
    # Part 1: 2x2x6 using 3 Strassen blocks
    # A_part: rows 0,1 of A? No, A is 2x5.
    # A is 2 rows, 5 cols.
    # We split A into A1 (2x2, cols 0,1) and A2 (2x3, cols 2,3,4).
    # B is 5 rows, 6 cols.
    # We split B into B1 (2x6, rows 0,1) and B2 (3x6, rows 2,3,4).
    
    # Block 1: A1 (2x2) * B1 (2x6)
    # B1 is split into 3 blocks of 2x2: B1_0 (cols 0,1), B1_1 (cols 2,3), B1_2 (cols 4,5)
    for k_block in range(3):
        # B1_k cols: 2*k_block, 2*k_block+1
        b_col_0 = 2*k_block
        b_col_1 = 2*k_block+1
        
        # Strassen terms for A1 * B1_k
        # A1 indices: (0,0)->0, (0,1)->1, (1,0)->5, (1,1)->6
        # B1_k indices: (0, b_col_0)->0*6+b_col_0, (0, b_col_1)->0*6+b_col_1
        #                 (1, b_col_0)->1*6+b_col_0, (1, b_col_1)->1*6+b_col_1
        
        a00, a01, a10, a11 = 0, 1, 5, 6
        b00, b01, b10, b11 = b_col_0, b_col_1, 6+b_col_0, 6+b_col_1
        
        # C indices for this block:
        # C is 6x2.
        # The result of A1 * B1_k is 2x2.
        # It contributes to C rows b_col_0, b_col_1 and cols 0, 1.
        # C index (row, col) -> row*2 + col
        c00, c01 = b_col_0*2 + 0, b_col_0*2 + 1
        c10, c11 = b_col_1*2 + 0, b_col_1*2 + 1
        
        # M1 = (A00+A11)(B00+B11)
        add_term([a00, a11], [1, 1], [b00, b11], [1, 1], 
                 [c00, c01, c10, c11], [1, 0, 0, 1]) # C00 += M1, C11 += M1
        
        # M2 = (A10+A11)B00
        add_term([a10, a11], [1, 1], [b00], [1], 
                 [c10, c11], [1, -1]) # C10 += M2, C11 -= M2
        
        # M3 = A00(B01-B11)
        add_term([a00], [1], [b01, b11], [1, -1], 
                 [c01, c11], [1, 1]) # C01 += M3, C11 += M3
        
        # M4 = A11(B10-B00)
        add_term([a11], [1], [b10, b00], [1, -1], 
                 [c00, c10], [1, 1]) # C00 += M4, C10 += M4
        
        # M5 = (A00+A01)B11
        add_term([a00, a01], [1, 1], [b11], [1], 
                 [c00, c01], [-1, 1]) # C00 -= M5, C01 += M5
        
        # M6 = (A10-A00)(B00+B01)
        add_term([a10, a00], [1, -1], [b00, b01], [1, 1], 
                 [c11], [1]) # C11 += M6
        
        # M7 = (A01-A11)(B10+B11)
        add_term([a01, a11], [1, -1], [b10, b11], [1, 1], 
                 [c00, c11], [1, 1]) # C00 += M7, C11 += M7

    # Part 2: 2x3x6 using 3 blocks of 2x3x2
    # A2 (2x3, cols 2,3,4)
    # B2 (3x6, rows 2,3,4)
    # B2 split into 3 blocks of 3x2: B2_0 (cols 0,1), B2_1 (cols 2,3), B2_2 (cols 4,5)
    for k_block in range(3):
        b_col_0 = 2*k_block
        b_col_1 = 2*k_block+1
        
        # A2 indices:
        # (0,2)->2, (0,3)->3, (0,4)->4
        # (1,2)->7, (1,3)->8, (1,4)->9
        a00, a01, a02 = 2, 3, 4
        a10, a11, a12 = 7, 8, 9
        
        # B2_k indices:
        # Rows 2,3,4. Cols b_col_0, b_col_1.
        # (2, b_col_0) -> 2*6+b_col_0
        # (2, b_col_1) -> 2*6+b_col_1
        # (3, b_col_0) -> 3*6+b_col_0
        # (3, b_col_1) -> 3*6+b_col_1
        # (4, b_col_0) -> 4*6+b_col_0
        # (4, b_col_1) -> 4*6+b_col_1
        b00, b01 = 12+b_col_0, 12+b_col_1
        b10, b11 = 18+b_col_0, 18+b_col_1
        b20, b21 = 24+b_col_0, 24+b_col_1
        
        # C indices:
        c00, c01 = b_col_0*2 + 0, b_col_0*2 + 1
        c10, c11 = b_col_1*2 + 0, b_col_1*2 + 1
        
        # Decomposition of 2x3x2 into 2x2x2 (7) + 2x1x2 (4)
        # A_sub1 = A2[:, 0:2] (2x2) -> a00, a01, a10, a11
        # B_sub1 = B2_k[0:2, :] (2x2) -> b00, b01, b10, b11
        # A_sub2 = A2[:, 2:3] (2x1) -> a02, a12
        # B_sub2 = B2_k[2:3, :] (1x2) -> b20, b21
        
        # 2x2x2 part (Strassen)
        # Same indices as before but mapped to this block's variables
        # M1 = (A00+A11)(B00+B11)
        add_term([a00, a11], [1, 1], [b00, b11], [1, 1], 
                 [c00, c01, c10, c11], [1, 0, 0, 1])
        
        # M2 = (A10+A11)B00
        add_term([a10, a11], [1, 1], [b00], [1], 
                 [c10, c11], [1, -1])
        
        # M3 = A00(B01-B11)
        add_term([a00], [1], [b01, b11], [1, -1], 
                 [c01, c11], [1, 1])
        
        # M4 = A11(B10-B00)
        add_term([a11], [1], [b10, b00], [1, -1], 
                 [c00, c10], [1, 1])
        
        # M5 = (A00+A01)B11
        add_term([a00, a01], [1, 1], [b11], [1], 
                 [c00, c01], [-1, 1])
        
        # M6 = (A10-A00)(B00+B01)
        add_term([a10, a00], [1, -1], [b00, b01], [1, 1], 
                 [c11], [1])
        
        # M7 = (A01-A11)(B10+B11)
        add_term([a01, a11], [1, -1], [b10, b11], [1, 1], 
                 [c00, c11], [1, 1])
        
        # 2x1x2 part (Rank 4)
        # A_sub2 = [a02; a12]
        # B_sub2 = [b20, b21]
        # C_contrib = A_sub2 * B_sub2
        # C00 += a02*b20
        # C01 += a02*b21
        # C10 += a12*b20
        # C11 += a12*b21
        
        add_term([a02], [1], [b20], [1], [c00], [1])
        add_term([a02], [1], [b21], [1], [c01], [1])
        add_term([a12], [1], [b20], [1], [c10], [1])
        add_term([a12], [1], [b21], [1], [c11], [1])

    # Convert terms to matrices
    rank = len(terms)
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    for i, (u, v, w) in enumerate(terms):
        U[:, i] = u
        V[:, i] = v
        W[:, i] = w
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
