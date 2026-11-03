import numpy as np

def construct():
    n, m, p = 4, 4, 7
    
    # Part 1: 4x4x4 with rank 49 (recursive Strassen)
    # Part 2: 4x4x3 with rank 48 (schoolbook)
    # Total rank = 97
    
    rank = 97
    U = [[0] * rank for _ in range(n * m)]  # 16 x 97
    V = [[0] * rank for _ in range(m * p)]  # 28 x 97
    W = [[0] * rank for _ in range(p * n)]  # 28 x 97
    
    # Strassen 2x2x2 decomposition (rank 7)
    # For multiplying 2x2 by 2x2
    # A = [[a0, a1], [a2, a3]], B = [[b0, b1], [b2, b3]]
    # C = [[c0, c1], [c2, c3]]
    # 
    # M1 = (a0+a3)(b0+b3)
    # M2 = (a2+a3)b0
    # M3 = a0(b1-b3)
    # M4 = a3(b2-b0)
    # M5 = (a0+a1)b3
    # M6 = (a2-a0)(b0+b1)
    # M7 = (a1-a3)(b2+b3)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # U2, V2, W2 for 2x2x2 rank-7
    # Indices: A: 0,1,2,3; B: 0,1,2,3; C: 0,1,2,3
    U2 = [
        [1, 1, 1, 0, 1, 0, 0],  # a0
        [0, 0, 0, 0, 1, 0, 1],  # a1
        [0, 1, 0, 0, 0, 1, 0],  # a2
        [1, 1, 0, 1, 0, 0, -1] # a3
    ]
    V2 = [
        [1, 1, 0, -1, 0, 1, 0], # b0
        [0, 0, 1, 0, 0, 1, 0],  # b1
        [0, 0, 0, 1, 0, 0, 1],  # b2
        [1, 0, -1, 0, 1, 0, 1]  # b3
    ]
    W2 = [
        [1, 0, 0, 0, -1, 0, 1], # C00
        [0, 0, 1, 0, 1, 0, 0],  # C01
        [0, 1, 0, 1, 0, 0, 0],  # C10
        [1, 0, 0, 0, 0, 1, 1]   # C11
    ]
    
    # For 4x4x4: split into 2x2 blocks
    # A blocks: A00 (rows 0-1, cols 0-1), A01 (rows 0-1, cols 2-3)
    #           A10 (rows 2-3, cols 0-1), A11 (rows 2-3, cols 2-3)
    # A indices in flat form:
    # A00: 0,1,4,5 (row-major: row0col0=0, row0col1=1, row1col0=4, row1col1=5)
    # A01: 2,3,6,7
    # A10: 8,9,12,13
    # A11: 10,11,14,15
    
    # B blocks for 4x4 part (cols 0-3):
    # B00: 0,1,4,5
    # B01: 2,3,6,7
    # B10: 8,9,12,13
    # B11: 10,11,14,15
    
    # C blocks for 4x4 part:
    # C00: 0,1,4,5
    # C01: 2,3,6,7
    # C10: 8,9,12,13
    # C11: 10,11,14,15
    
    # Block index to flat indices
    def get_block_indices(base, row_offset, col_offset, size=2):
        """Get flat indices for a 2x2 block starting at (row_offset, col_offset) in a matrix with 'base' stride"""
        indices = []
        for r in range(size):
            for c in range(size):
                indices.append((row_offset + r) * base + (col_offset + c))
        return indices
    
    # A block indices (base=4 for 4-wide matrix)
    A_blocks = {
        0: get_block_indices(4, 0, 0),  # A00: [0,1,4,5]
        1: get_block_indices(4, 0, 2),  # A01: [2,3,6,7]
        2: get_block_indices(4, 2, 0),  # A10: [8,9,12,13]
        3: get_block_indices(4, 2, 2),  # A11: [10,11,14,15]
    }
    
    # B block indices for 4x4 part (base=4 for first 4 cols)
    B_blocks_4 = {
        0: get_block_indices(4, 0, 0),  # B00
        1: get_block_indices(4, 0, 2),  # B01
        2: get_block_indices(4, 2, 0),  # B10
        3: get_block_indices(4, 2, 2),  # B11
    }
    
    # C block indices for 4x4 part (base=4 for first 4 cols)
    C_blocks_4 = {
        0: get_block_indices(4, 0, 0),  # C00
        1: get_block_indices(4, 0, 2),  # C01
        2: get_block_indices(4, 2, 0),  # C10
        3: get_block_indices(4, 2, 2),  # C11
    }
    
    # Strassen terms for 4x4x4
    # Each term: (A_block_combo, B_block_combo, C_block_combo, signs)
    # We need to express each of the 7 Strassen multiplications
    
    # Term 1: (A00+A11)(B00+B11) -> C00, C11
    # Term 2: (A10+A11)B00 -> C10
    # Term 3: A00(B01-B11) -> C01
    # Term 4: A11(B10-B00) -> C10
    # Term 5: (A00+A01)B11 -> C00, C01
    # Term 6: (A10-A00)(B00+B01) -> C11
    # Term 7: (A01-A11)(B10+B11) -> C00, C11
    
    # For each term, we have a "virtual" 2x2 A and 2x2 B that we multiply
    # The virtual A's entries are linear combinations of the block entries
    # The virtual B's entries are linear combinations of the block entries
    
    # Let me define the 7 terms more carefully
    # Each term has:
    #   - A coefficients: for each of 4 A block positions, a coefficient (0, 1, or -1)
    #   - B coefficients: for each of 4 B block positions, a coefficient (0, 1, or -1)
    #   - C output: which C blocks receive contributions and with what signs
    
    # Term 1: A = A00 + A11, B = B00 + B11
    #   A_coeff: [1, 0, 0, 1] (for blocks 0,1,2,3)
    #   B_coeff: [1, 0, 0, 1]
    #   C: C00 += M1, C11 += M1
    
    # Term 2: A = A10 + A11, B = B00
    #   A_coeff: [0, 0, 1, 1]
    #   B_coeff: [1, 0, 0, 0]
    #   C: C10 += M2
    
    # Term 3: A = A00, B = B01 - B11
    #   A_coeff: [1, 0, 0, 0]
    #   B_coeff: [0, 1, 0, -1]
    #   C: C01 += M3
    
    # Term 4: A = A11, B = B10 - B00
    #   A_coeff: [0, 0, 0, 1]
    #   B_coeff: [-1, 0, 1, 0]
    #   C: C10 += M4
    
    # Term 5: A = A00 + A01, B = B11
    #   A_coeff: [1, 1, 0, 0]
    #   B_coeff: [0, 0, 0, 1]
    #   C: C00 += M5, C01 += M5
    
    # Term 6: A = A10 - A00, B = B00 + B01
    #   A_coeff: [-1, 0, 1, 0]
    #   B_coeff: [1, 1, 0, 0]
    #   C: C11 += M6
    
    # Term 7: A = A01 - A11, B = B10 + B11
    #   A_coeff: [0, 1, 0, -1]
    #   B_coeff: [0, 0, 1, 1]
    #   C: C00 += M7, C11 += M7
    
    terms = [
        # (A_coeff, B_coeff, C_contribs)
        # C_contribs: list of (c_block, sign)
        ([1, 0, 0, 1], [1, 0, 0, 1], [(0, 1), (3, 1)]),   # Term 1
        ([0, 0, 1, 1], [1, 0, 0, 0], [(2, 1)]),           # Term 2
        ([1, 0, 0, 0], [0, 1, 0, -1], [(1, 1)]),          # Term 3
        ([0, 0, 0, 1], [-1, 0, 1, 0], [(2, 1)]),          # Term 4
        ([1, 1, 0, 0], [0, 0, 0, 1], [(0, 1), (1, 1)]),   # Term 5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [(3, 1)]),          # Term 6
        ([0, 1, 0, -1], [0, 0, 1, 1], [(0, 1), (3, 1)]),  # Term 7
    ]
    
    col = 0
    for term_idx in range(7):
        A_coeff, B_coeff, C_contribs = terms[term_idx]
        
        for sub in range(7):
            # This is sub-multiplication 'sub' of the 2x2x2 Strassen
            # U2[sub] gives the coefficients for A's 4 entries
            # V2[sub] gives the coefficients for B's 4 entries
            # W2[sub] gives the coefficients for C's 4 entries
            
            for a_block in range(4):
                if A_coeff[a_block] == 0:
                    continue
                a_indices = A_blocks[a_block]
                for local_idx in range(4):
                    if U2[local_idx][sub] == 0:
                        continue
                    val = A_coeff[a_block] * U2[local_idx][sub]
                    U[a_indices[local_idx]][col] += val
            
            for b_block in range(4):
                if B_coeff[b_block] == 0:
                    continue
                b_indices = B_blocks_4[b_block]
                for local_idx in range(4):
                    if V2[local_idx][sub] == 0:
                        continue
                    val = B_coeff[b_block] * V2[local_idx][sub]
                    V[b_indices[local_idx]][col] += val
            
            for c_block, c_sign in C_contribs:
                c_indices = C_blocks_4[c_block]
                for local_idx in range(4):
                    if W2[local_idx][sub] == 0:
                        continue
                    val = c_sign * W2[local_idx][sub]
                    W[c_indices[local_idx]][col] += val
            
            col += 1
    
    # Part 2: 4x4x3 schoolbook (cols 4,5,6 of B)
    for i in range(n):
        for j in range(m):
            for k in range(3):
                # B index: j*7 + (4+k)
                # C index: (4+k)*4 + i
                U[i * m + j][col] = 1
                V[j * p + (4 + k)][col] = 1
                W[(4 + k) * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
