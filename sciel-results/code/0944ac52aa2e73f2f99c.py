import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 28 using block Strassen
    rank = 28
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Helper to add a Strassen term
    # A_block is 2x2, B_block is 2x2
    # We need to map local indices to global indices
    
    # A is 2x4. Indices:
    # A00, A01, A02, A03
    # A10, A11, A12, A13
    # Global idx: i*4 + j
    
    # B is 4x4. Indices:
    # B00..B03
    # B10..B13
    # B20..B23
    # B30..B33
    # Global idx: i*4 + j
    
    # C is 2x4. Indices:
    # C00..C03
    # C10..C13
    # Global idx: k*2 + i (trace convention W[k*n+i])
    # So C[i,k] is at W[k*2 + i]
    
    def add_strassen_term(col_idx, a_indices, b_indices, c_coefficients):
        # a_indices: list of (global_idx, coeff)
        # b_indices: list of (global_idx, coeff)
        # c_coefficients: dict {global_c_idx: coeff}
        
        for idx, coeff in a_indices:
            U[idx, col_idx] += coeff
        for idx, coeff in b_indices:
            V[idx, col_idx] += coeff
        for idx, coeff in c_coefficients.items():
            W[idx, col_idx] += coeff

    # We have 4 products:
    # P1 = A1 B11 (A1 is A[:, 0:2], B11 is B[0:2, 0:2])
    # P2 = A1 B12 (A1 is A[:, 0:2], B12 is B[0:2, 2:4])
    # P3 = A2 B21 (A2 is A[:, 2:4], B21 is B[2:4, 0:2])
    # P4 = A2 B22 (A2 is A[:, 2:4], B22 is B[2:4, 2:4])
    
    # Strassen 7 mults for each 2x2 product.
    
    # Define local to global mappings for A blocks
    # A1: rows 0,1; cols 0,1
    # A1[0,0] -> A[0], A1[0,1] -> A[1], A1[1,0] -> A[4], A1[1,1] -> A[5]
    # A2: rows 0,1; cols 2,3
    # A2[0,0] -> A[2], A2[0,1] -> A[3], A2[1,0] -> A[6], A2[1,1] -> A[7]
    
    # B blocks
    # B11: rows 0,1; cols 0,1
    # B11[0,0] -> B[0], B11[0,1] -> B[1], B11[1,0] -> B[4], B11[1,1] -> B[5]
    # B12: rows 0,1; cols 2,3
    # B12[0,0] -> B[2], B12[0,1] -> B[3], B12[1,0] -> B[6], B12[1,1] -> B[7]
    # B21: rows 2,3; cols 0,1
    # B21[0,0] -> B[8], B21[0,1] -> B[9], B21[1,0] -> B[12], B21[1,1] -> B[13]
    # B22: rows 2,3; cols 2,3
    # B22[0,0] -> B[10], B22[0,1] -> B[11], B22[1,0] -> B[14], B22[1,1] -> B[15]
    
    # C blocks (Output)
    # C is 2x4.
    # C11: rows 0,1; cols 0,1
    # C11[0,0] -> C[0,0] -> W[0*2+0]=0
    # C11[0,1] -> C[0,1] -> W[1*2+0]=2
    # C11[1,0] -> C[1,0] -> W[0*2+1]=1
    # C11[1,1] -> C[1,1] -> W[1*2+1]=3
    # C12: rows 0,1; cols 2,3
    # C12[0,0] -> C[0,2] -> W[2*2+0]=4
    # C12[0,1] -> C[0,3] -> W[3*2+0]=6
    # C12[1,0] -> C[1,2] -> W[2*2+1]=5
    # C12[1,1] -> C[1,3] -> W[3*2+1]=7
    # C21: rows 0,1; cols 0,1 (from A2 B21)
    # Same as C11 indices
    # C22: rows 0,1; cols 2,3 (from A2 B22)
    # Same as C12 indices

    # Strassen coefficients for 2x2 product P = AB
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B11-B01)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # P00 = M1 + M4 - M5 + M7
    # P01 = M3 + M5
    # P10 = M2 + M4
    # P11 = M1 - M2 + M3 + M6
    
    # We will create 4 sets of 7 terms.
    
    def get_a_indices(block):
        if block == 0: # A1
            return {
                '00': 0, '01': 1, '10': 4, '11': 5
            }
        else: # A2
            return {
                '00': 2, '01': 3, '10': 6, '11': 7
            }

    def get_b_indices(block):
        if block == 0: # B11
            return {
                '00': 0, '01': 1, '10': 4, '11': 5
            }
        elif block == 1: # B12
            return {
                '00': 2, '01': 3, '10': 6, '11': 7
            }
        elif block == 2: # B21
            return {
                '00': 8, '01': 9, '10': 12, '11': 13
            }
        else: # B22
            return {
                '00': 10, '01': 11, '10': 14, '11': 15
            }

    def get_c_indices(col_block):
        # col_block 0 is cols 0,1; 1 is cols 2,3
        if col_block == 0:
            return {
                '00': 0, '01': 2, '10': 1, '11': 3
            }
        else:
            return {
                '00': 4, '01': 6, '10': 5, '11': 7
            }

    col = 0
    for a_block in [0, 1]:
        for b_block in [0, 1]:
            # Determine which B block
            # If a_block 0 (A1), B blocks are 0 (B11) and 1 (B12)
            # If a_block 1 (A2), B blocks are 2 (B21) and 3 (B22)
            if a_block == 0:
                b_idx = b_block
            else:
                b_idx = b_block + 2
            
            a_map = get_a_indices(a_block)
            b_map = get_b_indices(b_idx)
            
            # C column block is determined by B block column
            # B11, B21 are cols 0,1 -> c_block 0
            # B12, B22 are cols 2,3 -> c_block 1
            c_block = b_block
            c_map = get_c_indices(c_block)
            
            # 7 Strassen terms
            # M1
            add_strassen_term(col, 
                [(a_map['00'], 1), (a_map['11'], 1)],
                [(b_map['00'], 1), (b_map['11'], 1)],
                {c_map['00']: 1, c_map['11']: 1})
            col += 1
            # M2
            add_strassen_term(col,
                [(a_map['10'], 1), (a_map['11'], 1)],
                [(b_map['00'], 1)],
                {c_map['10']: 1})
            col += 1
            # M3
            add_strassen_term(col,
                [(a_map['00'], 1)],
                [(b_map['11'], 1), (b_map['01'], -1)],
                {c_map['01']: 1, c_map['11']: 1})
            col += 1
            # M4
            add_strassen_term(col,
                [(a_map['11'], 1)],
                [(b_map['10'], 1), (b_map['00'], -1)],
                {c_map['00']: 1, c_map['10']: 1})
            col += 1
            # M5
            add_strassen_term(col,
                [(a_map['00'], 1), (a_map['01'], 1)],
                [(b_map['11'], 1)],
                {c_map['00']: -1, c_map['01']: 1})
            col += 1
            # M6
            add_strassen_term(col,
                [(a_map['10'], 1), (a_map['00'], -1)],
                [(b_map['00'], 1), (b_map['01'], 1)],
                {c_map['11']: 1})
            col += 1
            # M7
            add_strassen_term(col,
                [(a_map['01'], 1), (a_map['11'], -1)],
                [(b_map['10'], 1), (b_map['11'], 1)],
                {c_map['00']: 1})
            col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
