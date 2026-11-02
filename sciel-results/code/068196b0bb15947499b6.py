import numpy as np

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # n=3, m=4, p=6
    # U: 12 x R, V: 24 x R, W: 18 x R
    
    rank = 66
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a rank-1 term
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        nonlocal col
        U[u_idx][col] += u_val
        V[v_idx][col] += v_val
        W[w_idx][col] += w_val
        col += 1

    # Row 1 of A (i=0)
    # A_0 is 1x4. B is 4x6.
    # Split B into B_0 (4x2), B_1 (4x2), B_2 (4x2).
    # C_0,0 = A_0 B_0 (1x2)
    # C_0,1 = A_0 B_1 (1x2)
    # C_0,2 = A_0 B_2 (1x2)
    
    # A_0 indices: j=0,1,2,3. i=0.
    # B_0 indices: j=0,1,2,3. k=0,1.
    # B_1 indices: j=0,1,2,3. k=2,3.
    # B_2 indices: j=0,1,2,3. k=4,5.
    
    # C_0,0: i=0, k=0,1.
    # Terms: A_0j * B_jk.
    for j in range(4):
        for k in range(2):
            # u_idx = i*m + j = 0*4 + j = j
            # v_idx = j*p + k = j*6 + k
            # w_idx = k*n + i = k*3 + 0 = k*3
            add_term(j, 1, j * 6 + k, 1, k * 3, 1)
            
    # C_0,1: i=0, k=2,3.
    for j in range(4):
        for k in range(2, 4):
            add_term(j, 1, j * 6 + k, 1, k * 3, 1)
            
    # C_0,2: i=0, k=4,5.
    for j in range(4):
        for k in range(4, 6):
            add_term(j, 1, j * 6 + k, 1, k * 3, 1)
            
    # Row 2,3 of A (i=1,2)
    # A_12 is 2x4.
    # Split A_12 into A_12_0 (2x2, cols 0,1) and A_12_1 (2x2, cols 2,3).
    # Split B into B_0 (4x2, cols 0,1), B_1 (4x2, cols 2,3), B_2 (4x2, cols 4,5).
    # Wait, B split was 4x2, 4x2, 4x2.
    # C_1,0 = A_12 B_0 (2x2).
    # C_1,1 = A_12 B_1 (2x2).
    # C_1,2 = A_12 B_2 (2x2).
    
    # For each C_1,x, we have A_12 (2x4) * B_x (4x2).
    # A_12 = [A_left (2x2), A_right (2x2)].
    # B_x = [B_top (2x2); B_bottom (2x2)].
    # C_1,x = A_left B_top + A_right B_bottom.
    # Each product is 2x2x2. Use Strassen.
    
    def add_strassen_block(a_row_offset, a_col_offset, b_row_offset, b_col_offset):
        nonlocal col
        # a_row_offset: 0 for i=1, 1 for i=2.
        # a_col_offset: 0 for j=0,1; 2 for j=2,3.
        # b_row_offset: 0 for j=0,1; 2 for j=2,3.
        # b_col_offset: 0 for k=0,1; 2 for k=2,3; 4 for k=4,5.
        
        # A submatrix indices:
        # i in {a_row_offset, a_row_offset+1}
        # j in {a_col_offset, a_col_offset+1}
        # u_idx = i*4 + j
        
        # B submatrix indices:
        # j in {b_row_offset, b_row_offset+1}
        # k in {b_col_offset, b_col_offset+1}
        # v_idx = j*6 + k
        
        # C submatrix indices:
        # i in {a_row_offset, a_row_offset+1}
        # k in {b_col_offset, b_col_offset+1}
        # w_idx = k*3 + i
        
        # Strassen M1..M7
        # M1 = (a+d)(e+h)
        # a: i=a_row_offset, j=a_col_offset
        # d: i=a_row_offset+1, j=a_col_offset+1
        # e: j=b_row_offset, k=b_col_offset
        # h: j=b_row_offset+1, k=b_col_offset+1
        
        # We add 7 terms.
        # Each term is u * v * w.
        # u is linear in A, v is linear in B, w is 1.
        
        # M1: u = e_a + e_d, v = e_e + e_h
        add_term(a_row_offset * 4 + a_col_offset, 1, b_row_offset * 6 + b_col_offset, 1, (b_col_offset) * 3 + a_row_offset, 1)
        # Wait, the w index depends on the specific output cell.
        # Strassen computes the whole 2x2 C.
        # The rank-1 terms contribute to ALL 4 cells of C.
        # So for M1, we need to add to W for all 4 (i,k) pairs?
        # No, the rank-1 term is a single tensor.
        # The tensor is sum of u_i v_j w_k.
        # If u has non-zero at two indices, and v at two, and w at four?
        # No, w must be a vector.
        # In Strassen, M1 contributes to C11, C22.
        # So w should have 1s at the positions corresponding to C11 and C22?
        # Yes.
        
        # Let's re-verify the Strassen decomposition in terms of rank-1 tensors.
        # T = sum M_k (u_k \otimes v_k \otimes w_k).
        # For M1, u1 = e_a + e_d. v1 = e_e + e_h.
        # w1 must be such that u1 \otimes v1 \otimes w1 gives the contribution.
        # Contribution to C11: 1. To C22: 1. To C12, C21: 0.
        # So w1 should have 1 at (i=0, k=0) and (i=1, k=1) relative to the block.
        # Global w indices:
        # (i=a_row_offset, k=b_col_offset) -> w_idx1
        # (i=a_row_offset+1, k=b_col_offset+1) -> w_idx2
        # w1 has 1 at w_idx1 and w_idx2.
        
        # So for M1, we add:
        # U: 1 at a, 1 at d.
        # V: 1 at e, 1 at h.
        # W: 1 at w1, 1 at w2.
        
        # This is a single rank-1 term in the global tensor?
        # Yes, because U, V, W are vectors.
        # U has two non-zeros. V has two. W has two.
        # The outer product is sum of 4 terms.
        # But it counts as 1 rank.
        
        # So I just need to add the coefficients to U, V, W for this column.
        
        # M1
        U[a_row_offset * 4 + a_col_offset][col] += 1
        U[(a_row_offset + 1) * 4 + (a_col_offset + 1)][col] += 1
        V[b_row_offset * 6 + b_col_offset][col] += 1
        V[(b_row_offset + 1) * 6 + (b_col_offset + 1)][col] += 1
        W[b_col_offset * 3 + a_row_offset][col] += 1
        W[(b_col_offset + 1) * 3 + (a_row_offset + 1)][col] += 1
        col += 1
        
        # M2 = (c+d)e
        # c: i=a_row_offset+1, j=a_col_offset
        # d: i=a_row_offset+1, j=a_col_offset+1
        # e: j=b_row_offset, k=b_col_offset
        # Contribution to C21: 1.
        U[(a_row_offset + 1) * 4 + a_col_offset][col] += 1
        U[(a_row_offset + 1) * 4 + (a_col_offset + 1)][col] += 1
        V[b_row_offset * 6 + b_col_offset][col] += 1
        W[b_col_offset * 3 + (a_row_offset + 1)][col] += 1
        col += 1
        
        # M3 = a(f-h)
        # a: i=a_row_offset, j=a_col_offset
        # f: j=b_row_offset, k=b_col_offset+1
        # h: j=b_row_offset+1, k=b_col_offset+1
        # Contribution to C12: 1.
        U[a_row_offset * 4 + a_col_offset][col] += 1
        V[b_row_offset * 6 + (b_col_offset + 1)][col] += 1
        V[(b_row_offset + 1) * 6 + (b_col_offset + 1)][col] += -1
        W[(b_col_offset + 1) * 3 + a_row_offset][col] += 1
        col += 1
        
        # M4 = d(g-e)
        # d: i=a_row_offset+1, j=a_col_offset+1
        # g: j=b_row_offset+1, k=b_col_offset
        # e: j=b_row_offset, k=b_col_offset
        # Contribution to C11: 1, C21: 1.
        U[(a_row_offset + 1) * 4 + (a_col_offset + 1)][col] += 1
        V[(b_row_offset + 1) * 6 + b_col_offset][col] += 1
        V[b_row_offset * 6 + b_col_offset][col] += -1
        W[b_col_offset * 3 + a_row_offset][col] += 1
        W[b_col_offset * 3 + (a_row_offset + 1)][col] += 1
        col += 1
        
        # M5 = (a+b)h
        # a: i=a_row_offset, j=a_col_offset
        # b: i=a_row_offset, j=a_col_offset+1
        # h: j=b_row_offset+1, k=b_col_offset+1
        # Contribution to C11: -1, C12: 1.
        U[a_row_offset * 4 + a_col_offset][col] += 1
        U[a_row_offset * 4 + (a_col_offset + 1)][col] += 1
        V[(b_row_offset + 1) * 6 + (b_col_offset + 1)][col] += 1
        W[b_col_offset * 3 + a_row_offset][col] += -1
        W[(b_col_offset + 1) * 3 + a_row_offset][col] += 1
        col += 1
        
        # M6 = (c-a)(e+f)
        # c: i=a_row_offset+1, j=a_col_offset
        # a: i=a_row_offset, j=a_col_offset
        # e: j=b_row_offset, k=b_col_offset
        # f: j=b_row_offset, k=b_col_offset+1
        # Contribution to C22: 1.
        U[(a_row_offset + 1) * 4 + a_col_offset][col] += 1
        U[a_row_offset * 4 + a_col_offset][col] += -1
        V[b_row_offset * 6 + b_col_offset][col] += 1
        V[b_row_offset * 6 + (b_col_offset + 1)][col] += 1
        W[(b_col_offset + 1) * 3 + (a_row_offset + 1)][col] += 1
        col += 1
        
        # M7 = (b-d)(g+h)
        # b: i=a_row_offset, j=a_col_offset+1
        # d: i=a_row_offset+1, j=a_col_offset+1
        # g: j=b_row_offset+1, k=b_col_offset
        # h: j=b_row_offset+1, k=b_col_offset+1
        # Contribution to C11: 1.
        U[a_row_offset * 4 + (a_col_offset + 1)][col] += 1
        U[(a_row_offset + 1) * 4 + (a_col_offset + 1)][col] += -1
        V[(b_row_offset + 1) * 6 + b_col_offset][col] += 1
        V[(b_row_offset + 1) * 6 + (b_col_offset + 1)][col] += 1
        W[b_col_offset * 3 + a_row_offset][col] += 1
        col += 1

    # Call for the 6 blocks in row 2
    # a_row_offset is 0 (for i=1) or 1 (for i=2)?
    # Global i=1 is offset 0 in the 2x4 block.
    # Global i=2 is offset 1.
    # a_col_offset 0 (j=0,1) or 2 (j=2,3).
    # b_row_offset 0 (j=0,1) or 2 (j=2,3).
    # b_col_offset 0 (k=0,1), 2 (k=2,3), 4 (k=4,5).
    
    # Block C_1,0: i=1,2; k=0,1.
    # A part: j=0,1 and j=2,3.
    # B part: j=0,1 and j=2,3.
    add_strassen_block(0, 0, 0, 0) # A_left, B_top
    add_strassen_block(0, 2, 2, 0) # A_right, B_bottom
    
    # Block C_1,1: i=1,2; k=2,3.
    add_strassen_block(0, 0, 0, 2)
    add_strassen_block(0, 2, 2, 2)
    
    # Block C_1,2: i=1,2; k=4,5.
    add_strassen_block(0, 0, 0, 4)
    add_strassen_block(0, 2, 2, 4)
    
    return {"U": U, "V": V, "W": W}
