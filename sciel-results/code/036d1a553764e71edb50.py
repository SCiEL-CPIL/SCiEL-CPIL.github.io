"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 7 (Strassen) * 27 (Schoolbook 3x3) = 189
    rank = 189
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Block offsets for 2x2 block structure of 3x3 blocks
    # (row_offset, col_offset)
    block_offsets = {
        11: (0, 0),
        12: (0, 3),
        21: (3, 0),
        22: (3, 3)
    }
    
    # Strassen terms definition
    # Each term: (left_blocks, right_blocks, output_blocks)
    # left_blocks: list of (block_id, sign)
    # right_blocks: list of (block_id, sign)
    # output_blocks: list of (block_id, sign)
    
    strassen_terms = [
        # M1 = (A11 + A22)(B11 + B22)
        ([ (11, 1), (22, 1) ], [ (11, 1), (22, 1) ], [ (11, 1), (22, 1) ]),
        # M2 = (A21 + A22)B11
        ([ (21, 1), (22, 1) ], [ (11, 1) ], [ (21, 1), (22, -1) ]),
        # M3 = A11(B12 - B22)
        ([ (11, 1) ], [ (12, 1), (22, -1) ], [ (12, 1), (22, 1) ]),
        # M4 = A22(B21 - B11)
        ([ (22, 1) ], [ (21, 1), (11, -1) ], [ (11, 1), (21, 1) ]),
        # M5 = (A11 + A12)B22
        ([ (11, 1), (12, 1) ], [ (22, 1) ], [ (11, -1), (12, 1) ]),
        # M6 = (A21 - A11)(B11 + B12)
        ([ (21, 1), (11, -1) ], [ (11, 1), (12, 1) ], [ (22, 1) ]),
        # M7 = (A12 - A22)(B21 + B22)
        ([ (12, 1), (22, -1) ], [ (21, 1), (22, 1) ], [ (11, 1) ])
    ]
    
    col_idx = 0
    for left_blocks, right_blocks, out_blocks in strassen_terms:
        # Iterate over 3x3x3 scalar multiplications for the block product
        for i in range(3):
            for k in range(3):
                for j in range(3):
                    # This iteration computes the contribution of L_ik * R_kj to (L*R)_ij
                    c = col_idx
                    col_idx += 1
                    
                    # Update U (Left factor)
                    for block_id, sign in left_blocks:
                        r_off, c_off = block_offsets[block_id]
                        row_A = r_off + i
                        col_A = c_off + k
                        U[row_A * m + col_A][c] += sign
                    
                    # Update V (Right factor)
                    for block_id, sign in right_blocks:
                        r_off, c_off = block_offsets[block_id]
                        row_B = r_off + k
                        col_B = c_off + j
                        V[row_B * p + col_B][c] += sign
                    
                    # Update W (Output factor)
                    # Convention: W[k*n + i, r] corresponds to C[i, k]
                    # Here, the result is added to C blocks.
                    # The local index in the block is (i, j).
                    # So global row is r_off_C + i, global col is c_off_C + j.
                    # W index is global_col * n + global_row.
                    for block_id, sign in out_blocks:
                        r_off, c_off = block_offsets[block_id]
                        row_C = r_off + i
                        col_C = c_off + j
                        W[col_C * n + row_C][c] += sign
                        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
