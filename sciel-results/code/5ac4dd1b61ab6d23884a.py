import time
import numpy as np

def construct():
    n, m, p = 4, 4, 9
    
    # Total rank: 
    # 4x4x4 (cols 0-3) -> Rank 49
    # 4x4x4 (cols 4-7) -> Rank 49
    # 4x4x1 (col 8)    -> Rank 16
    # Total = 114
    
    rank = 114
    U = [[0] * rank for _ in range(n * m)] # 16 x 114
    V = [[0] * rank for _ in range(m * p)] # 36 x 114
    W = [[0] * rank for _ in range(p * n)] # 36 x 114
    
    # Helper to add a 4x4x4 Strassen block
    # A_cols: list of 4 column indices for A (actually A is fixed 4x4, so indices are 0-3)
    # B_cols: list of 4 column indices for B
    # C_cols: list of 4 column indices for C (which correspond to B's cols)
    # Offset in the rank dimension
    def add_4x4x4_strassen(start_rank, b_col_offset, c_col_offset):
        # b_col_offset: 0 for first 4x4x4, 4 for second
        # c_col_offset: same as b_col_offset
        
        # Define 2x2 Strassen U2, V2, W2
        # Local indices for 2x2:
        # A: 0,1,2,3
        # B: 0,1,2,3
        # C: 0,1,2,3 (W convention: j*2+i)
        
        U2 = [[0]*7 for _ in range(4)]
        V2 = [[0]*7 for _ in range(4)]
        W2 = [[0]*7 for _ in range(4)]
        
        # r=0: M1 = (A00+A11)(B00+B11)
        U2[0][0] = 1; U2[3][0] = 1
        V2[0][0] = 1; V2[3][0] = 1
        W2[0][0] = 1; W2[3][0] = 1
        
        # r=1: M2 = (A10+A11)B00
        U2[2][1] = 1; U2[3][1] = 1
        V2[0][1] = 1
        W2[2][1] = 1; W2[3][1] = -1
        
        # r=2: M3 = A00(B01-B11)
        U2[0][2] = 1
        V2[1][2] = 1; V2[3][2] = -1
        W2[1][2] = 1
        
        # r=3: M4 = A11(B10-B00)
        U2[3][3] = 1
        V2[2][3] = 1; V2[0][3] = -1
        W2[0][3] = -1; W2[1][3] = 1
        
        # r=4: M5 = (A00+A01)B11
        U2[0][4] = 1; U2[1][4] = 1
        V2[3][4] = 1
        W2[3][4] = -1
        
        # r=5: M6 = (A10-A00)(B00+B01)
        U2[2][5] = 1; U2[0][5] = -1
        V2[0][5] = 1; V2[1][5] = 1
        W2[0][5] = -1; W2[2][5] = 1
        
        # r=6: M7 = (A01-A11)(B10+B11)
        U2[1][6] = 1; U2[3][6] = -1
        V2[2][6] = 1; V2[3][6] = 1
        W2[0][6] = 1; W2[3][6] = 1
        
        # Now expand to 4x4x4
        # A blocks:
        # A11 (rows 0-1, cols 0-1): Global A indices 0,1,4,5
        # A12 (rows 0-1, cols 2-3): Global A indices 2,3,6,7
        # A21 (rows 2-3, cols 0-1): Global A indices 8,9,12,13
        # A22 (rows 2-3, cols 2-3): Global A indices 10,11,14,15
        
        # B blocks (for this 4x4x4 subproblem):
        # B11 (rows 0-1, cols 0-3): Global B indices 0,1,4,5
        # B12 (rows 0-1, cols 4-7): Global B indices 2,3,6,7  <-- Wait, B is 4x9.
        # The 4x4x4 subproblem uses B columns b_col_offset .. b_col_offset+3
        # So if b_col_offset=0, B cols are 0,1,2,3.
        # If b_col_offset=4, B cols are 4,5,6,7.
        
        # C blocks (for this 4x4x4 subproblem):
        # C uses same column offset as B.
        
        # Mapping local 2x2 indices to global 4x4 indices
        # Local A index la (0..3) -> Global A index
        # la=0 (A11_00) -> 0
        # la=1 (A11_01) -> 1
        # la=2 (A11_10) -> 4
        # la=3 (A11_11) -> 5
        # la=4 (A12_00) -> 2  (Note: U2 is defined for 2x2 block. We need to map 4 blocks)
        
        # Actually, it is simpler to iterate over the 4 blocks of A and B.
        # The 7 Strassen multiplications are applied to the 4x4 blocks.
        # M_k = A_block * B_block.
        # A is 4x4. B is 4x4 (submatrix).
        # The Strassen formulas combine the 4 2x2 sub-blocks of A and B.
        
        # Let's define the mapping for A (4x4) to the 4 local 2x2 blocks used in Strassen.
        # Strassen uses 4 sub-blocks of A: A11, A12, A21, A22.
        # Each is 2x2.
        # We need to map the 16 elements of A to the coefficients in U.
        # U is 16x49.
        
        # For each Strassen step r (0..6):
        # We have 7 scalar multiplications of 2x2 matrices? No.
        # Strassen computes 7 matrix products of size 2x2.
        # Each 2x2 product has rank 7.
        # So for each of the 7 Strassen steps, we add 7 rank-1 terms.
        # Total 49 terms.
        
        # For a specific Strassen step r:
        # The "A-part" is a linear combination of the 4 sub-blocks A11..A22.
        # The "B-part" is a linear combination of the 4 sub-blocks B11..B22.
        # The result is a 2x2 matrix which is placed into the corresponding C sub-blocks.
        
        # This is getting complicated to map manually. 
        # Alternative: Use the fact that Rank(4x4x4) <= 49 is established by applying Rank(2x2x2)=7 to the 7 Strassen steps.
        # We can construct the 49 vectors by iterating r=0..6 and s=0..6 (inner Strassen rank).
        
        # Let's define the 2x2 Strassen matrices U2, V2, W2 as above.
        # For a given Strassen step r (0..6):
        # The A-submatrix is $A_r = \sum_{block} u_{block} A_{block}$.
        # The B-submatrix is $B_r = \sum_{block} v_{block} B_{block}$.
        # The product $C_r = A_r B_r$ is a 2x2 matrix.
        # This $C_r$ contributes to the final C blocks with coefficients from the outer Strassen formulas.
        
        # Wait, the outer Strassen formulas determine WHICH sub-blocks of A and B are combined.
        # M1 = (A11+A22)(B11+B22). Here A_r = A11+A22, B_r = B11+B22.
        # M2 = (A21+A22)B11. Here A_r = A21+A22, B_r = B11.
        # ...
        
        # So for each r=0..6, we identify the coefficients for A11, A12, A21, A22 and B11, B12, B21, B22.
        # Then we use the inner 2x2 Strassen (rank 7) to decompose the product of these 2x2 matrices.
        
        # Outer Strassen Coefficients:
        # r=0: A: {11:1, 22:1}, B: {11:1, 22:1}
        # r=1: A: {21:1, 22:1}, B: {11:1}
        # r=2: A: {11:1}, B: {12:1, 22:-1}
        # r=3: A: {22:1}, B: {21:1, 11:-1}
        # r=4: A: {11:1, 12:1}, B: {22:1}
        # r=5: A: {21:1, 11:-1}, B: {11:1, 12:1}
        # r=6: A: {12:1, 22:-1}, B: {21:1, 22:1}
        
        # Inner Strassen (2x2x2) U2, V2, W2 are defined above.
        # U2 is 4x7. V2 is 4x7. W2 is 4x7.
        # Local indices 0,1,2,3 correspond to the 4 entries of the 2x2 matrix.
        
        # Mapping Global A indices to Local A indices for each block:
        # A11 (2x2): Global 0,1,4,5 -> Local 0,1,2,3
        # A12 (2x2): Global 2,3,6,7 -> Local 0,1,2,3
        # A21 (2x2): Global 8,9,12,13 -> Local 0,1,2,3
        # A22 (2x2): Global 10,11,14,15 -> Local 0,1,2,3
        
        # Mapping Global B indices (for this 4x4x4 subproblem) to Local B indices:
        # B is 4x9. We are using columns b_col_offset .. b_col_offset+3.
        # B11 (2x2): Rows 0-1, Cols b_off .. b_off+1.
        #    Global B indices: k*9 + j.
        #    k=0, j=b_off -> 0*9+b_off
        #    k=0, j=b_off+1 -> 0*9+b_off+1
        #    k=1, j=b_off -> 1*9+b_off
        #    k=1, j=b_off+1 -> 1*9+b_off+1
        # B12 (2x2): Rows 0-1, Cols b_off+2 .. b_off+3.
        # B21 (2x2): Rows 2-3, Cols b_off .. b_off+1.
        # B22 (2x2): Rows 2-3, Cols b_off+2 .. b_off+3.
        
        # Mapping Global C indices to Local C indices:
        # C is 4x9. Columns c_col_offset .. c_col_offset+3.
        # C11 (2x2): Rows 0-1, Cols c_off .. c_off+1.
        #    Global C indices: j*4 + i.
        #    i=0, j=c_off -> c_off*4+0
        #    i=0, j=c_off+1 -> (c_off+1)*4+0
        #    i=1, j=c_off -> c_off*4+1
        #    i=1, j=c_off+1 -> (c_off+1)*4+1
        #    Note: W convention is W[k*n+i, r]. So index is j*n + i.
        # C12 (2x2): Rows 0-1, Cols c_off+2 .. c_off+3.
        # C21 (2x2): Rows 2-3, Cols c_off .. c_off+1.
        # C22 (2x2): Rows 2-3, Cols c_off+2 .. c_off+3.

        # Outer Strassen C-combination:
        # C11 = M1 - M4 - M6 + M7
        # C12 = M3 + M4
        # C21 = M2 + M6
        # C22 = M1 - M2 - M5 + M7
        
        # For each r=0..6, we have an "inner" decomposition of the 2x2 product.
        # The inner decomposition produces 7 scalar terms.
        # Each scalar term corresponds to a rank-1 tensor.
        # We need to add these to U, V, W.
        
        # Let's pre-calculate the global indices for A, B, C blocks.
        a_blocks = {
            '11': [0, 1, 4, 5],
            '12': [2, 3, 6, 7],
            '21': [8, 9, 12, 13],
            '22': [10, 11, 14, 15]
        }
        
        b_blocks = {}
        # B11
        b_blocks['11'] = [0*9 + b_col_offset, 0*9 + b_col_offset + 1, 1*9 + b_col_offset, 1*9 + b_col_offset + 1]
        # B12
        b_blocks['12'] = [0*9 + b_col_offset + 2, 0*9 + b_col_offset + 3, 1*9 + b_col_offset + 2, 1*9 + b_col_offset + 3]
        # B21
        b_blocks['21'] = [2*9 + b_col_offset, 2*9 + b_col_offset + 1, 3*9 + b_col_offset, 3*9 + b_col_offset + 1]
        # B22
        b_blocks['22'] = [2*9 + b_col_offset + 2, 2*9 + b_col_offset + 3, 3*9 + b_col_offset + 2, 3*9 + b_col_offset + 3]
        
        c_blocks = {}
        # C11
        c_blocks['11'] = [c_col_offset*4 + 0, (c_col_offset+1)*4 + 0, c_col_offset*4 + 1, (c_col_offset+1)*4 + 1]
        # C12
        c_blocks['12'] = [(c_col_offset+2)*4 + 0, (c_col_offset+3)*4 + 0, (c_col_offset+2)*4 + 1, (c_col_offset+3)*4 + 1]
        # C21
        c_blocks['21'] = [c_col_offset*4 + 2, (c_col_offset+1)*4 + 2, c_col_offset*4 + 3, (c_col_offset+1)*4 + 3]
        # C22
        c_blocks['22'] = [(c_col_offset+2)*4 + 2, (c_col_offset+3)*4 + 2, (c_col_offset+2)*4 + 3, (c_col_offset+3)*4 + 3]
        
        # Outer Strassen Definitions
        # r, A_coeffs, B_coeffs, C_coeffs
        # A_coeffs: dict block->coeff
        # B_coeffs: dict block->coeff
        # C_coeffs: dict block->coeff (for the RESULT of the 2x2 product)
        
        outer_strassen = [
            ({'11': 1, '22': 1}, {'11': 1, '22': 1}, {'11': 1, '12': 0, '21': 0, '22': 1}), # M1
            ({'21': 1, '22': 1}, {'11': 1}, {'11': 0, '12': 0, '21': 1, '22': -1}),       # M2
            ({'11': 1}, {'12': 1, '22': -1}, {'11': 0, '12': 1, '21': 0, '22': 0}),       # M3
            ({'22': 1}, {'21': 1, '11': -1}, {'11': -1, '12': 1, '21': 0, '22': 0}),      # M4
            ({'11': 1, '12': 1}, {'22': 1}, {'11': 0, '12': 0, '21': 0, '22': -1}),       # M5
            ({'21': 1, '11': -1}, {'11': 1, '12': 1}, {'11': -1, '12': 0, '21': 1, '22': 0}), # M6
            ({'12': 1, '22': -1}, {'21': 1, '22': 1}, {'11': 1, '12': 0, '21': 0, '22': 1})  # M7
        ]
        
        # Inner Strassen U2, V2, W2 (4x7)
        # U2[i][s], V2[j][s], W2[k][s]
        # i,j,k are local 2x2 indices 0..3
        
        for r_idx, (a_c, b_c, c_c) in enumerate(outer_strassen):
            # For each of the 7 inner Strassen steps s
            for s in range(7):
                current_rank = start_rank + r_idx * 7 + s
                
                # Construct U vector for this rank
                # U is 16x1. We need to fill the 16 entries.
                # The A-part is a linear combination of A blocks.
                # A_r = sum_{block} a_c[block] * A_block
                # The inner Strassen step s uses U2[local_a][s] to combine the 4 entries of A_r.
                # So Global A index = block_global_idx[local_a]
                # Coeff = a_c[block] * U2[local_a][s]
                
                u_vals = [0]*16
                for block_name, coeff_a in a_c.items():
                    global_indices = a_blocks[block_name]
                    for local_idx in range(4):
                        g_idx = global_indices[local_idx]
                        u_vals[g_idx] += coeff_a * U2[local_idx][s]
                
                # Construct V vector
                v_vals = [0]*36
                for block_name, coeff_b in b_c.items():
                    global_indices = b_blocks[block_name]
                    for local_idx in range(4):
                        g_idx = global_indices[local_idx]
                        v_vals[g_idx] += coeff_b * V2[local_idx][s]
                
                # Construct W vector
                # The result C_r is a 2x2 matrix.
                # The inner Strassen step s produces a scalar which is the dot product of A_r and B_r components.
                # Wait, the inner Strassen decomposes the 2x2x2 tensor.
                # The output of the inner Strassen is a 2x2 matrix C_r.
                # C_r[local_c] = sum_s W2[local_c][s] * (A_r . B_r)_s
                # So the contribution to Global C is:
                # For each block_name in c_c:
                #   For each local_c in 0..3:
                #     Global C index = c_blocks[block_name][local_c]
                #     Coeff = c_c[block_name] * W2[local_c][s]
                
                w_vals = [0]*36
                for block_name, coeff_c in c_c.items():
                    global_indices = c_blocks[block_name]
                    for local_idx in range(4):
                        g_idx = global_indices[local_idx]
                        w_vals[g_idx] += coeff_c * W2[local_idx][s]
                
                # Add to U, V, W
                for i in range(16):
                    if u_vals[i] != 0:
                        U[i][current_rank] = u_vals[i]
                for j in range(36):
                    if v_vals[j] != 0:
                        V[j][current_rank] = v_vals[j]
                for k in range(36):
                    if w_vals[k] != 0:
                        W[k][current_rank] = w_vals[k]

    # Call for first 4x4x4 (cols 0-3)
    add_4x4x4_strassen(0, 0, 0)
    
    # Call for second 4x4x4 (cols 4-7)
    add_4x4x4_strassen(49, 4, 4)
    
    # Call for 4x4x1 (col 8) - Schoolbook Rank 16
    start_rank = 98
    for i in range(4):
        for k in range(4):
            for j in range(0, 1): # Only col 8
                U[i*4 + k][start_rank] = 1
                V[k*9 + 8][start_rank] = 1
                W[8*4 + i][start_rank] = 1
                start_rank += 1
    
    return {"U": U, "V": V, "W": W}
