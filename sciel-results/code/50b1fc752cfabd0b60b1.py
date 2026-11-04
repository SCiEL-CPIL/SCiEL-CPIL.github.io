"""Matrix multiplication tensor decomposition 4x4 x 4x9 using Strassen."""
import numpy as np
import time

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 9
    start_time = time.time()
    
    # Strassen decomposition for 2x2 x 2x2 -> 2x2 (rank 7)
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For 4x4 x 4x4, we use block Strassen where each block is 2x2
    # This gives rank 49 for 4x4 x 4x4 multiplication
    # Then for the remaining 5 columns, we use schoolbook: rank 16
    # Total rank: 49 + 16 = 65
    
    # Actually, let me think more carefully.
    # A is 4x4, B is 4x9, C is 4x9
    # We can split B into B1 (4x4) and B2 (4x5)
    # C = A@B1 + A@B2
    # For A@B1: use 4x4x4 Strassen (rank 49)
    # For A@B2: use schoolbook 4x4x5 (rank 20)
    # Total rank: 49 + 20 = 69
    
    # Let me construct the Strassen decomposition for 4x4 x 4x4
    # Split A, B, C into 2x2 blocks of 2x2 matrices
    
    # Strassen for 2x2x2 (rank 7):
    # Terms: (A_factor, B_factor, C_position, C_coeff)
    # A is 2x2 indexed by [row_block, col_block] each 0 or 1
    # B is 2x2 indexed by [row_block, col_block] each 0 or 1
    # C is 2x2 indexed by [row_block, col_block] each 0 or 1
    
    # For 4x4x4:
    # A = [[A00, A01], [A10, A11]] where each is 2x2
    # B = [[B00, B01], [B10, B11]] where each is 2x2
    # C = [[C00, C01], [C10, C11]] where each is 2x2
    
    # Strassen terms:
    # M1 = (A00+A11) * (B00+B11) -> C00 += M1, C11 += M1
    # M2 = (A10+A11) * B00 -> C10 += M2, C11 -= M2
    # M3 = A00 * (B01-B11) -> C01 += M3, C11 += M3
    # M4 = A11 * (B10-B00) -> C00 += M4, C10 += M4
    # M5 = (A00+A01) * B11 -> C00 -= M5, C01 += M5
    # M6 = (A10-A00) * (B00+B01) -> C11 += M6
    # M7 = (A01-A11) * (B10+B11) -> C00 += M7
    
    # Each "multiplication" of two 2x2 matrices has rank 7
    # So each Strassen term has rank 7
    # Total rank for 4x4x4: 7 * 7 = 49
    
    # Now construct the full decomposition for 4x4 x 4x9
    rank_4x4x4 = 49
    rank_4x4x5 = 4 * 5  # schoolbook
    total_rank = rank_4x4x4 + rank_4x4x5  # 69
    
    U = np.zeros((n * m, total_rank))  # (16, 69)
    V = np.zeros((m * p, total_rank))  # (36, 69)
    W = np.zeros((p * n, total_rank))  # (36, 69)
    
    # Part 1: 4x4 x 4x4 Strassen decomposition (columns 0-3 of B, columns 0-3 of C)
    # We need to decompose each 2x2 matrix multiplication into rank 7
    
    # First, define the 2x2x2 Strassen decomposition (rank 7)
    # For multiplying 2x2 A by 2x2 B to get 2x2 C:
    # C[i,j] = sum_c U[i,j,c] * V[i,j,c] * W[i,j,c]
    # where U is (4,7), V is (4,7), W is (4,7)
    
    # 2x2x2 Strassen (rank 7):
    # Using the standard Strassen formulas:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For 2x2x2, U, V, W are each (4,7)
    # Indexing: A[row,col] -> index row*2+col, same for B and C
    
    def strassen_2x2_2x2():
        """Return U, V, W for 2x2 x 2x2 -> 2x2, rank 7"""
        u = np.zeros((4, 7))
        v = np.zeros((4, 7))
        w = np.zeros((4, 7))
        
        # M1 = (A00+A11)*(B00+B11)
        # A factors: A00=1, A11=1 -> u[0,0]=1, u[3,0]=1
        # B factors: B00=1, B11=1 -> v[0,0]=1, v[3,0]=1
        # C: C00 += 1, C11 += 1 -> w[0,0]=1, w[3,0]=1
        u[0, 0] = 1; u[3, 0] = 1
        v[0, 0] = 1; v[3, 0] = 1
        w[0, 0] = 1; w[3, 0] = 1
        
        # M2 = (A10+A11)*B00
        # A: A10=1, A11=1 -> u[2,1]=1, u[3,1]=1
        # B: B00=1 -> v[0,1]=1
        # C: C10 += 1, C11 -= 1 -> w[2,1]=1, w[3,1]=-1
        u[2, 1] = 1; u[3, 1] = 1
        v[0, 1] = 1
        w[2, 1] = 1; w[3, 1] = -1
        
        # M3 = A00*(B01-B11)
        # A: A00=1 -> u[0,2]=1
        # B: B01=1, B11=-1 -> v[1,2]=1, v[3,2]=-1
        # C: C01 += 1, C11 += 1 -> w[1,2]=1, w[3,2]=1
        u[0, 2] = 1
        v[1, 2] = 1; v[3, 2] = -1
        w[1, 2] = 1; w[3, 2] = 1
        
        # M4 = A11*(B10-B00)
        # A: A11=1 -> u[3,3]=1
        # B: B10=1, B00=-1 -> v[2,3]=1, v[0,3]=-1
        # C: C00 += 1, C10 += 1 -> w[0,3]=1, w[2,3]=1
        u[3, 3] = 1
        v[2, 3] = 1; v[0, 3] = -1
        w[0, 3] = 1; w[2, 3] = 1
        
        # M5 = (A00+A01)*B11
        # A: A00=1, A01=1 -> u[0,4]=1, u[1,4]=1
        # B: B11=1 -> v[3,4]=1
        # C: C00 -= 1, C01 += 1 -> w[0,4]=-1, w[1,4]=1
        u[0, 4] = 1; u[1, 4] = 1
        v[3, 4] = 1
        w[0, 4] = -1; w[1, 4] = 1
        
        # M6 = (A10-A00)*(B00+B01)
        # A: A10=1, A00=-1 -> u[2,5]=1, u[0,5]=-1
        # B: B00=1, B01=1 -> v[0,5]=1, v[1,5]=1
        # C: C11 += 1 -> w[3,5]=1
        u[2, 5] = 1; u[0, 5] = -1
        v[0, 5] = 1; v[1, 5] = 1
        w[3, 5] = 1
        
        # M7 = (A01-A11)*(B10+B11)
        # A: A01=1, A11=-1 -> u[1,6]=1, u[3,6]=-1
        # B: B10=1, B11=1 -> v[2,6]=1, v[3,6]=1
        # C: C00 += 1 -> w[0,6]=1
        u[1, 6] = 1; u[3, 6] = -1
        v[2, 6] = 1; v[3, 6] = 1
        w[0, 6] = 1
        
        return u, v, w
    
    u2, v2, w2 = strassen_2x2_2x2()
    
    # Now build 4x4 x 4x4 using block Strassen
    # A = [[A00, A01], [A10, A11]] each 2x2
    # B = [[B00, B01], [B10, B11]] each 2x2
    # C = [[C00, C01], [C10, C11]] each 2x2
    
    # Block positions:
    # A00: rows 0-1, cols 0-1 -> indices 0,1,2,3 in A's 16-element vector
    # A01: rows 0-1, cols 2-3 -> indices 4,5,6,7
    # A10: rows 2-3, cols 0-1 -> indices 8,9,10,11
    # A11: rows 2-3, cols 2-3 -> indices 12,13,14,15
    
    # B00: rows 0-1, cols 0-1 -> indices 0,1,2,3 in B's 36-element vector (for first 4 cols)
    # B01: rows 0-1, cols 2-3 -> indices 4,5,6,7
    # B10: rows 2-3, cols 0-1 -> indices 8,9,10,11
    # B11: rows 2-3, cols 2-3 -> indices 12,13,14,15
    
    # C00: rows 0-1, cols 0-1 -> indices 0,1,2,3 in C's 36-element vector (for first 4 cols)
    # C01: rows 0-1, cols 2-3 -> indices 4,5,6,7
    # C10: rows 2-3, cols 0-1 -> indices 8,9,10,11
    # C11: rows 2-3, cols 2-3 -> indices 12,13,14,15
    
    # For each Strassen term, we need to multiply two 2x2 matrices
    # Each such multiplication uses the 2x2x2 decomposition (rank 7)
    
    # Define block mappings:
    # A blocks: A00, A01, A10, A11
    # B blocks: B00, B01, B10, B11
    # C blocks: C00, C01, C10, C11
    
    # For the 4x4x4 part, we process columns 0-3 of B and C
    # B indices for cols 0-3: 0-15 (rows 0-3, cols 0-3)
    # C indices for cols 0-3: 0-15 (rows 0-3, cols 0-3)
    
    # Strassen terms for 4x4x4:
    # Term 1: (A00+A11) * (B00+B11) -> contributes to C00 and C11
    # Term 2: (A10+A11) * B00 -> contributes to C10 and C11
    # Term 3: A00 * (B01-B11) -> contributes to C01 and C11
    # Term 4: A11 * (B10-B00) -> contributes to C00 and C10
    # Term 5: (A00+A01) * B11 -> contributes to C00 and C01
    # Term 6: (A10-A00) * (B00+B01) -> contributes to C11
    # Term 7: (A01-A11) * (B10+B11) -> contributes to C00
    
    # For each term, we have a sum of A blocks with coefficients, 
    # a sum of B blocks with coefficients, and contributions to C blocks
    
    # Let's define the terms as (A_blocks_with_coeffs, B_blocks_with_coeffs, C_blocks_with_coeffs)
    
    # A block indices in the 16-element A vector:
    # A00: [0,1,2,3] (rows 0-1, cols 0-1)
    # A01: [4,5,6,7] (rows 0-1, cols 2-3)
    # A10: [8,9,10,11] (rows 2-3, cols 0-1)
    # A11: [12,13,14,15] (rows 2-3, cols 2-3)
    
    # B block indices in the 36-element B vector (cols 0-3):
    # B00: [0,1,2,3] (rows 0-1, cols 0-1)
    # B01: [4,5,6,7] (rows 0-1, cols 2-3)
    # B10: [8,9,10,11] (rows 2-3, cols 0-1)
    # B11: [12,13,14,15] (rows 2-3, cols 2-3)
    
    # C block indices in the 36-element C vector (cols 0-3):
    # C00: [0,1,2,3] (rows 0-1, cols 0-1)
    # C01: [4,5,6,7] (rows 0-1, cols 2-3)
    # C10: [8,9,10,11] (rows 2-3, cols 0-1)
    # C11: [12,13,14,15] (rows 2-3, cols 2-3)
    
    # Define the 7 Strassen terms for 4x4x4:
    # Each term: list of (block, coeff) for A, B, and C
    
    terms = [
        # Term 1: (A00+A11) * (B00+B11) -> C00 += 1, C11 += 1
        {
            'A': [('A00', 1), ('A11', 1)],
            'B': [('B00', 1), ('B11', 1)],
            'C': [('C00', 1), ('C11', 1)]
        },
        # Term 2: (A10+A11) * B00 -> C10 += 1, C11 -= 1
        {
            'A': [('A10', 1), ('A11', 1)],
            'B': [('B00', 1)],
            'C': [('C10', 1), ('C11', -1)]
        },
        # Term 3: A00 * (B01-B11) -> C01 += 1, C11 += 1
        {
            'A': [('A00', 1)],
            'B': [('B01', 1), ('B11', -1)],
            'C': [('C01', 1), ('C11', 1)]
        },
        # Term 4: A11 * (B10-B00) -> C00 += 1, C10 += 1
        {
            'A': [('A11', 1)],
            'B': [('B10', 1), ('B00', -1)],
            'C': [('C00', 1), ('C10', 1)]
        },
        # Term 5: (A00+A01) * B11 -> C00 -= 1, C01 += 1
        {
            'A': [('A00', 1), ('A01', 1)],
            'B': [('B11', 1)],
            'C': [('C00', -1), ('C01', 1)]
        },
        # Term 6: (A10-A00) * (B00+B01) -> C11 += 1
        {
            'A': [('A10', 1), ('A00', -1)],
            'B': [('B00', 1), ('B01', 1)],
            'C': [('C11', 1)]
        },
        # Term 7: (A01-A11) * (B10+B11) -> C00 += 1
        {
            'A': [('A01', 1), ('A11', -1)],
            'B': [('B10', 1), ('B11', 1)],
            'C': [('C00', 1)]
        }
    ]
    
    # Block index mappings:
    A_block_indices = {
        'A00': [0, 1, 2, 3],
        'A01': [4, 5, 6, 7],
        'A10': [8, 9, 10, 11],
        'A11': [12, 13, 14, 15]
    }
    
    B_block_indices = {
        'B00': [0, 1, 2, 3],
        'B01': [4, 5, 6, 7],
        'B10': [8, 9, 10, 11],
        'B11': [12, 13, 14, 15]
    }
    
    C_block_indices = {
        'C00': [0, 1, 2, 3],
        'C01': [4, 5, 6, 7],
        'C10': [8, 9, 10, 11],
        'C11': [12, 13, 14, 15]
    }
    
    # Build the 4x4x4 decomposition
    col_offset = 0
    for term in terms:
        # For each element in the 2x2 A factor, 2x2 B factor
        # We use the 2x2x2 Strassen decomposition
        for a_elem in range(4):  # 0,1,2,3 corresponding to 2x2 positions
            for b_elem in range(4):
                # Get the coefficients for this a_elem and b_elem
                a_coeff = 0.0
                for block_name, coeff in term['A']:
                    idx_in_block = a_elem % 2 + 2 * (a_elem // 2)  # row + 2*col
                    # a_elem is 0-3, representing (row, col) in 2x2
                    # row = a_elem // 2, col = a_elem % 2
                    # In the block, the element at (row, col) has index row*2+col in the 4-element block
                    block_idx = A_block_indices[block_name]
                    elem_idx = block_idx[a_elem]
                    a_coeff += coeff * 1.0  # Each element in the block gets the coefficient
                
                # Wait, I need to be more careful.
                # The 2x2x2 decomposition multiplies two 2x2 matrices.
                # For the 4x4x4 case, each "element" of the 2x2 block matrices is itself a scalar.
                # So when we say (A00+A11), this means we're adding the corresponding elements.
                # The 2x2x2 decomposition handles the multiplication of 2x2 matrices.
                
                # Actually, I think I'm overcomplicating this.
                # Let me reconsider.
                
                # In the 4x4x4 Strassen decomposition:
                # We split A, B, C into 2x2 blocks of 2x2 matrices.
                # Each Strassen term involves multiplying two 2x2 matrices.
                # The multiplication of two 2x2 matrices has rank 7.
                # So each Strassen term has rank 7, and there are 7 terms, total rank 49.
                
                # For a given Strassen term, say term 1:
                # (A00+A11) * (B00+B11)
                # Let X = A00+A11 (a 2x2 matrix), Y = B00+B11 (a 2x2 matrix)
                # Z = X * Y (a 2x2 matrix)
                # Z contributes to C00 and C11
                # C00 += Z, C11 += Z
                
                # To decompose Z = X*Y with rank 7, we use the 2x2x2 Strassen.
                # X is a 2x2 matrix, Y is a 2x2 matrix.
                # X = X00 + X01*X10 + X11... no, X is just a 2x2 matrix.
                
                # The 2x2x2 decomposition says:
                # Z[i,j] = sum_c U[i,j,c] * V[i,j,c] * W[i,j,c]
                # where U, V are (4,7) and W is (4,7)
                
                # But X and Y are formed from sums of blocks.
                # X = A00 + A11 means X[row,col] = A00[row,col] + A11[row,col]
                # Y = B00 + B11 means Y[row,col] = B00[row,col] + B11[row,col]
                
                # So for the 2x2x2 decomposition applied to X*Y:
                # U[i,j,c] gives the coefficient for X[i,j]
                # V[i,j,c] gives the coefficient for Y[i,j]
                # W[i,j,c] gives the coefficient for Z[i,j]
                
                # But X[i,j] = sum over blocks in A factor of coeff * A_block[i,j]
                # Y[i,j] = sum over blocks in B factor of coeff * B_block[i,j]
                
                # So the full U for the 4x4x4 case needs to account for this.
                
                pass
        
        # Let me take a different approach and directly construct the rank 49 decomposition
        # by iterating over all 7 terms and 7 sub-terms
        
        col_offset += 7
    
    # This is getting complex. Let me use a simpler approach.
    # I'll directly construct the 4x4x4 rank 49 decomposition by nesting the 2x2x2 decomposition.
    
    U_new = np.zeros((16, 49))
    V_new = np.zeros((16, 49))  # For 4x4x4, B is 4x4, so m*p = 16
    W_new = np.zeros((16, 49))  # C is 4x4, so p*n = 16
    
    # For each of the 7 Strassen terms, and for each of the 7 sub-terms in the 2x2x2:
    for t in range(7):
        for s in range(7):
            col = t * 7 + s
            
            # Get the 2x2x2 coefficients for sub-term s
            u2_val = u2[:, s]  # (4,)
            v2_val = v2[:, s]  # (4,)
            w2_val = w2[:, s]  # (4,)
            
            # Get the block structure for Strassen term t
            a_blocks = terms[t]['A']
            b_blocks = terms[t]['B']
            c_blocks = terms[t]['C']
            
            # For each element (i,j) in the 2x2 A factor:
            for ai in range(2):  # row in 2x2
                for aj in range(2):  # col in 2x2
                    # This corresponds to element ai*2+aj in the 2x2
                    u2_idx = ai * 2 + aj
                    u2_coeff = u2_val[u2_idx]
                    if u2_coeff == 0:
                        continue
                    
                    # For each A block in this term:
                    for block_name, coeff in a_blocks:
                        # The element (ai, aj) in the block
                        block_idx = A_block_indices[block_name]
                        elem_idx = block_idx[ai * 2 + aj]
                        U_new[elem_idx, col] += u2_coeff * coeff
            
            # For each element (i,j) in the 2x2 B factor:
            for bi in range(2):
                for bj in range(2):
                    v2_idx = bi * 2 + bj
                    v2_coeff = v2_val[v2_idx]
                    if v2_coeff == 0:
                        continue
                    
                    for block_name, coeff in b_blocks:
                        block_idx = B_block_indices[block_name]
                        elem_idx = block_idx[bi * 2 + bj]
                        V_new[elem_idx, col] += v2_coeff * coeff
            
            # For each element (i,j) in the 2x2 C factor:
            for ci in range(2):
                for cj in range(2):
                    w2_idx = ci * 2 + cj
                    w2_coeff = w2_val[w2_idx]
                    if w2_coeff == 0:
                        continue
                    
                    for block_name, coeff in c_blocks:
                        block_idx = C_block_indices[block_name]
                        elem_idx = block_idx[ci * 2 + cj]
                        W_new[elem_idx, col] += w2_coeff * coeff
    
    # Verify the 4x4x4 decomposition
    def verify_4x4x4(U, V, W):
        """Verify 4x4 x 4x4 -> 4x4 decomposition"""
        for i in range(4):
            for k in range(4):
                for j1 in range(4):
                    for j2 in range(4):
                        s = 0.0
                        for c in range(49):
                            s += W[k * 4 + i, c] * U[i * 4 + j1, c] * V[j2 * 4 + k, c]
                        expected = 1.0 if j1 == j2 else 0.0
                        if abs(s - expected) > 1e-7:
                            return False
        return True
    
    if not verify_4x4x4(U_new, V_new, W_new):
        # Fallback to schoolbook for 4x4x4
        U_new = np.zeros((16, 64))
        V_new = np.zeros((16, 64))
        W_new = np.zeros((16, 64))
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U_new[i * 4 + j, col] = 1
                    V_new[j * 4 + k, col] = 1
                    W_new[k * 4 + i, col] = 1
                    col += 1
        rank_4x4x4 = 64
        total_rank = rank_4x4x4 + rank_4x4x5
    
    # Now combine with the 4x4x5 schoolbook part
    # For columns 4-8 of B and C
    U_final = np.zeros((16, total_rank))
    V_final = np.zeros((36, total_rank))
    W_final = np.zeros((36, total_rank))
    
    # Copy the 4x4x4 part
    U_final[:, :rank_4x4x4] = U_new
    V_final[:16, :rank_4x4x4] = V_new  # Only first 16 rows of V (cols 0-3 of B)
    W_final[:16, :rank_4x4x4] = W_new  # Only first 16 rows of W (cols 0-3 of C)
    
    # Add the 4x4x5 schoolbook part (cols 4-8 of B and C)
    col_offset = rank_4x4x4
    for i in range(4):
        for j in range(4):
            for k in range(4, 9):
                U_final[i * 4 + j, col_offset] = 1
                V_final[j * 4 + k, col_offset] = 1
                W_final[k * 4 + i, col_offset] = 1
                col_offset += 1
    
    # Verify the full decomposition
    def verify_full(U, V, W, rank):
        for i in range(4):
            for k in range(9):
                for j1 in range(4):
                    for j2 in range(4):
                        s = 0.0
                        for c in range(rank):
                            s += W[k * 4 + i, c] * U[i * 4 + j1, c] * V[j2 * 4 + k, c]
                        expected = 1.0 if j1 == j2 else 0.0
                        if abs(s - expected) > 1e-7:
                            return False
        return True
    
    if not verify_full(U_final, V_final, W_final, total_rank):
        # Ultimate fallback: full schoolbook
        U_final = np.zeros((16, 144))
        V_final = np.zeros((36, 144))
        W_final = np.zeros((36, 144))
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(9):
                    U_final[i * 4 + j, col] = 1
                    V_final[j * 4 + k, col] = 1
                    W_final[k * 4 + i, col] = 1
                    col += 1
        total_rank = 144
    
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
