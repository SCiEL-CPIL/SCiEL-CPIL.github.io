import numpy as np

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=7
    
    # We will construct the decomposition for the two blocks:
    # Block 1: A_L (2x2) * B_L (2x7)
    # Block 2: A_R (2x2) * B_R (2x7)
    
    # Total rank will be 50.
    # U shape: (8, 50)
    # V shape: (28, 50)
    # W shape: (14, 50)
    
    rank = 50
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Helper to add a term
    # a_coeffs: dict or list of (i, j, coeff) for A
    # b_coeffs: dict or list of (j, k, coeff) for B
    # c_coeffs: dict or list of (i, k, coeff) for C
    # Note: A indices are (i, j) with i in [0, n), j in [0, m)
    # B indices are (j, k) with j in [0, m), k in [0, p)
    # C indices are (i, k) with i in [0, n), k in [0, p)
    # But W index is k * n + i.
    
    def add_term(a_coeffs, b_coeffs, c_coeffs):
        nonlocal col
        for (i, j), val in a_coeffs.items():
            U[i * m + j, col] += val
        for (j, k), val in b_coeffs.items():
            V[j * p + k, col] += val
        for (i, k), val in c_coeffs.items():
            # W index is k * n + i
            W[k * n + i, col] += val
        col += 1

    # Strassen terms for 2x2x2
    # Local indices: a00, a01, a10, a11
    # b00, b01, b10, b11
    # c00, c01, c10, c11
    
    # M1 = (a00 + a11) * (b00 + b11)
    # c00 += M1, c11 += M1
    # M2 = (a10 + a11) * b00
    # c10 += M2, c11 -= M2
    # M3 = a00 * (b01 - b11)
    # c01 += M3, c11 += M3
    # M4 = a11 * (b10 - b00)
    # c00 += M4, c10 += M4
    # M5 = (a00 + a01) * b11
    # c00 -= M5, c01 += M5
    # M6 = (a10 - a00) * (b00 + b01)
    # c11 += M6
    # M7 = (a01 - a11) * (b10 + b11)
    # c00 += M7
    
    def add_strassen_block(a_base_i, a_base_j, b_base_j, b_base_k, c_base_i, c_base_k):
        # a_base_i: row offset in A (0 or 1)
        # a_base_j: col offset in A (0 or 1)
        # b_base_j: row offset in B (0 or 1)
        # b_base_k: col offset in B (0..6)
        # c_base_i: row offset in C (0 or 1)
        # c_base_k: col offset in C (0..6)
        
        # M1
        a1 = { (a_base_i, a_base_j): 1, (a_base_i+1, a_base_j+1): 1 }
        b1 = { (b_base_j, b_base_k): 1, (b_base_j+1, b_base_k+1): 1 }
        c1 = { (c_base_i, c_base_k): 1, (c_base_i+1, c_base_k+1): 1 }
        add_term(a1, b1, c1)
        
        # M2
        a2 = { (a_base_i+1, a_base_j): 1, (a_base_i+1, a_base_j+1): 1 }
        b2 = { (b_base_j, b_base_k): 1 }
        c2 = { (c_base_i+1, c_base_k): 1, (c_base_i+1, c_base_k+1): -1 }
        add_term(a2, b2, c2)
        
        # M3
        a3 = { (a_base_i, a_base_j): 1 }
        b3 = { (b_base_j, b_base_k+1): 1, (b_base_j+1, b_base_k+1): -1 }
        c3 = { (c_base_i, c_base_k+1): 1, (c_base_i+1, c_base_k+1): 1 }
        add_term(a3, b3, c3)
        
        # M4
        a4 = { (a_base_i+1, a_base_j+1): 1 }
        b4 = { (b_base_j+1, b_base_k): 1, (b_base_j, b_base_k): -1 }
        c4 = { (c_base_i, c_base_k): 1, (c_base_i+1, c_base_k): 1 }
        add_term(a4, b4, c4)
        
        # M5
        a5 = { (a_base_i, a_base_j): 1, (a_base_i, a_base_j+1): 1 }
        b5 = { (b_base_j+1, b_base_k+1): 1 }
        c5 = { (c_base_i, c_base_k): -1, (c_base_i, c_base_k+1): 1 }
        add_term(a5, b5, c5)
        
        # M6
        a6 = { (a_base_i+1, a_base_j): 1, (a_base_i, a_base_j): -1 }
        b6 = { (b_base_j, b_base_k): 1, (b_base_j, b_base_k+1): 1 }
        c6 = { (c_base_i+1, c_base_k+1): 1 }
        add_term(a6, b6, c6)
        
        # M7
        a7 = { (a_base_i, a_base_j+1): 1, (a_base_i+1, a_base_j+1): -1 }
        b7 = { (b_base_j+1, b_base_k): 1, (b_base_j+1, b_base_k+1): 1 }
        c7 = { (c_base_i, c_base_k): 1 }
        add_term(a7, b7, c7)

    # Helper for 2x2x1 (Schoolbook)
    # A (2x2) * B (2x1) -> C (2x1)
    # C0 = A00*B0 + A01*B1
    # C1 = A10*B0 + A11*B1
    # 4 terms
    def add_2x2x1_block(a_base_i, a_base_j, b_base_j, b_base_k, c_base_i, c_base_k):
        # Term 1: A00 * B0 -> C0
        add_term({(a_base_i, a_base_j): 1}, {(b_base_j, b_base_k): 1}, {(c_base_i, c_base_k): 1})
        # Term 2: A01 * B1 -> C0
        add_term({(a_base_i, a_base_j+1): 1}, {(b_base_j+1, b_base_k): 1}, {(c_base_i, c_base_k): 1})
        # Term 3: A10 * B0 -> C1
        add_term({(a_base_i+1, a_base_j): 1}, {(b_base_j, b_base_k): 1}, {(c_base_i+1, c_base_k): 1})
        # Term 4: A11 * B1 -> C1
        add_term({(a_base_i+1, a_base_j+1): 1}, {(b_base_j+1, b_base_k): 1}, {(c_base_i+1, c_base_k): 1})

    # Block 1: A_L (cols 0,1 of A) * B_L (rows 0,1 of B)
    # A_L is at rows 0,1 of A.
    # B_L is at rows 0,1 of B.
    # Result C is at rows 0,1 of C.
    
    # B_L is 2x7. Split into 2x2, 2x2, 2x2, 2x1.
    # B_L cols 0,1 -> 2x2
    add_strassen_block(0, 0, 0, 0, 0, 0)
    
    # B_L cols 2,3 -> 2x2
    add_strassen_block(0, 0, 0, 2, 0, 2)
    
    # B_L cols 4,5 -> 2x2
    add_strassen_block(0, 0, 0, 4, 0, 4)
    
    # B_L cols 6 -> 2x1
    add_2x2x1_block(0, 0, 0, 6, 0, 6)

    # Block 2: A_R (cols 2,3 of A) * B_R (rows 2,3 of B)
    # A_R is at rows 0,1 of A, cols 2,3.
    # B_R is at rows 2,3 of B.
    # Result C is at rows 0,1 of C.
    
    # B_R cols 0,1 -> 2x2
    add_strassen_block(0, 2, 2, 0, 0, 0)
    
    # B_R cols 2,3 -> 2x2
    add_strassen_block(0, 2, 2, 2, 0, 2)
    
    # B_R cols 4,5 -> 2x2
    add_strassen_block(0, 2, 2, 4, 0, 4)
    
    # B_R cols 6 -> 2x1
    add_2x2x1_block(0, 2, 2, 6, 0, 6)

    # Check if col == 50
    # 7 * 3 + 4 = 25 per block. 2 blocks = 50.
    
    return {"U": U, "V": V, "W": W}
