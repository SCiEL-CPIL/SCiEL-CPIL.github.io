import numpy as np

def construct():
    n, m, p = 2, 3, 8
    
    # We decompose the multiplication tensor for 2x3 by 3x8.
    # We split the 8 columns of B into 4 blocks of 2 columns each.
    # For each block, we multiply a 2x3 matrix by a 3x2 matrix to get a 2x2 result.
    # The rank of the 2x3 by 3x2 multiplication tensor is 11.
    # This can be achieved by decomposing the 3x2 matrix into a 2x2 part and a 1x2 part.
    # The 2x2 part is handled by Strassen's algorithm (rank 7).
    # The 1x2 part is handled by outer products (rank 4).
    # Total rank per block = 7 + 4 = 11.
    # Total rank for 4 blocks = 4 * 11 = 44.
    
    rank = 44
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen's algorithm terms for 2x2 by 2x2
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # C = A * B
    # Strassen terms:
    # m1 = (a00 + a11) * (b00 + b11)
    # m2 = (a10 + a11) * b00
    # m3 = a00 * (b01 - b11)
    # m4 = a11 * (b10 - b00)
    # m5 = (a00 + a01) * b11
    # m6 = (a10 - a00) * (b00 + b01)
    # m7 = (a01 - a11) * (b10 + b11)
    # C00 = m1 + m4 - m5 + m7
    # C01 = m3 + m5
    # C10 = m2 + m4
    # C11 = m1 - m2 + m3 + m6
    
    # Strassen coefficients for U (A part), V (B part), W (C part)
    # U indices: 0=a00, 1=a01, 2=a10, 3=a11
    # V indices: 0=b00, 1=b01, 2=b10, 3=b11
    # W indices: 0=C00, 1=C01, 2=C10, 3=C11
    
    strassen_terms = [
        # (u_indices, u_vals, v_indices, v_vals, w_indices, w_vals)
        ((0, 3), (1, 1), (0, 3), (1, 1), (0,), (1,)),  # m1
        ((2, 3), (1, 1), (0,), (1,), (2,), (1,)),       # m2
        ((0,), (1,), (1, 3), (1, -1), (1,), (1,)),      # m3
        ((3,), (1,), (2, 0), (1, -1), (0, 2), (1, 1)),  # m4
        ((0, 1), (1, 1), (3,), (1,), (0, 1), (-1, 1)),  # m5
        ((2, 0), (1, -1), (0, 1), (1, 1), (3,), (1,)),  # m6
        ((1, 3), (1, -1), (2, 3), (1, 1), (0,), (1,)),  # m7
    ]
    
    # Outer product terms for the 1x2 part (row 2 of A times 2x2 part of B)
    # A row 2: [a02, a12]
    # B part: [[b02, b03], [b12, b13]]
    # C00 += a02 * b02
    # C01 += a02 * b03
    # C10 += a12 * b02
    # C11 += a12 * b03
    
    # U indices: 4=a02, 5=a12
    # V indices: 4=b02, 5=b03, 6=b12, 7=b13
    # W indices: 0=C00, 1=C01, 2=C10, 3=C11
    
    outer_terms = [
        ((4,), (1,), (4,), (1,), (0,), (1,)),  # a02 * b02
        ((4,), (1,), (5,), (1,), (1,), (1,)),  # a02 * b03
        ((5,), (1,), (4,), (1,), (2,), (1,)),  # a12 * b02
        ((5,), (1,), (5,), (1,), (3,), (1,)),  # a12 * b03
    ]
    
    all_terms = strassen_terms + outer_terms
    
    for block in range(4):
        # Block b corresponds to columns 2*b and 2*b+1 of B
        # U indices are the same for all blocks: 0..5
        # V indices are shifted by block:
        #   b00 -> 0*8 + 2*block = 2*block
        #   b01 -> 0*8 + 2*block + 1 = 2*block + 1
        #   b10 -> 1*8 + 2*block = 8 + 2*block
        #   b11 -> 1*8 + 2*block + 1 = 8 + 2*block + 1
        #   b02 -> 0*8 + 2*block = 2*block (wait, no, b02 is the third row, first col of the 3x2 block)
        #   Let's re-index.
        #   The 3x2 block is:
        #   [[B[0, 2b], B[0, 2b+1]],
        #    [B[1, 2b], B[1, 2b+1]],
        #    [B[2, 2b], B[2, 2b+1]]]
        #   So:
        #   b00 = B[0, 2b] -> V index 0*8 + 2b = 2b
        #   b01 = B[0, 2b+1] -> V index 0*8 + 2b + 1 = 2b + 1
        #   b10 = B[1, 2b] -> V index 1*8 + 2b = 8 + 2b
        #   b11 = B[1, 2b+1] -> V index 1*8 + 2b + 1 = 8 + 2b + 1
        #   b02 = B[2, 2b] -> V index 2*8 + 2b = 16 + 2b
        #   b03 = B[2, 2b+1] -> V index 2*8 + 2b + 1 = 16 + 2b + 1
        #   b12, b13 are not used in the outer product terms, only b02, b03 (which are the third row)
        #   Wait, in my outer_terms, I used V indices 4, 5 for b02, b03.
        #   Let's map the local V indices (0..7) to global V indices.
        #   Local V index 0 (b00) -> Global 2b
        #   Local V index 1 (b01) -> Global 2b + 1
        #   Local V index 2 (b10) -> Global 8 + 2b
        #   Local V index 3 (b11) -> Global 8 + 2b + 1
        #   Local V index 4 (b02) -> Global 16 + 2b
        #   Local V index 5 (b03) -> Global 16 + 2b + 1
        
        v_shift = [2*block, 2*block+1, 8+2*block, 8+2*block+1, 16+2*block, 16+2*block+1]
        
        # W indices are shifted by block:
        #   C00 -> W index (2*block)*2 + 0 = 4*block
        #   C01 -> W index (2*block+1)*2 + 0 = 4*block + 2
        #   C10 -> W index (2*block)*2 + 1 = 4*block + 1
        #   C11 -> W index (2*block+1)*2 + 1 = 4*block + 3
        
        w_shift = [4*block, 4*block+2, 4*block+1, 4*block+3]
        
        base_rank = block * 11
        
        for term in all_terms:
            u_indices, u_vals, v_indices, v_vals, w_indices, w_vals = term
            
            for ui, uv in zip(u_indices, u_vals):
                U[ui, base_rank + 0] += uv  # This is wrong, each term is a separate rank-1 component
            
            # Actually, each term in `all_terms` is not a single rank-1 component, but a set of rank-1 components.
            # Let's re-structure. Each term in `all_terms` defines a set of rank-1 components.
            # For example, m1 = (a00 + a11) * (b00 + b11) * C00.
            # This is not a single rank-1 tensor. It's a sum of 4 rank-1 tensors:
            # a00*b00*C00 + a00*b11*C00 + a11*b00*C00 + a11*b11*C00.
            # But Strassen's algorithm combines these into a single scalar multiplication.
            # The tensor rank decomposition requires expressing the tensor as a sum of rank-1 tensors.
            # A rank-1 tensor is U_col * V_col * W_col.
            # Strassen's algorithm reduces the number of scalar multiplications, but does it reduce the tensor rank?
            # Yes, the tensor rank of <2,2,2> is 7, which is less than 8.
            # The 7 rank-1 tensors correspond to the 7 Strassen products.
            # Each Strassen product is a scalar multiplication of a linear combination of A's, a linear combination of B's, and a linear combination of C's.
            # So each Strassen product corresponds to one rank-1 tensor in the decomposition.
            # The U_col is the vector of coefficients of the linear combination of A's.
            # The V_col is the vector of coefficients of the linear combination of B's.
            # The W_col is the vector of coefficients of the linear combination of C's.
            
            # So for m1 = (a00 + a11) * (b00 + b11) * C00:
            # U_col has 1 at a00, 1 at a11.
            # V_col has 1 at b00, 1 at b11.
            # W_col has 1 at C00.
            # This is one rank-1 tensor.
            
            # My `strassen_terms` structure is correct. Each term is one rank-1 tensor.
            # The `u_indices` and `u_vals` define the U_col.
            # The `v_indices` and `v_vals` define the V_col.
            # The `w_indices` and `w_vals` define the W_col.
            
            # So I need to fill in one column of U, V, W for each term.
            
            pass
    
    # Let's redo the filling.
    rank = 44
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    for block in range(4):
        v_shift = [2*block, 2*block+1, 8+2*block, 8+2*block+1, 16+2*block, 16+2*block+1]
        w_shift = [4*block, 4*block+2, 4*block+1, 4*block+3]
        
        base_rank = block * 11
        
        for term_idx, term in enumerate(all_terms):
            u_indices, u_vals, v_indices, v_vals, w_indices, w_vals = term
            r = base_rank + term_idx
            
            for ui, uv in zip(u_indices, u_vals):
                U[ui, r] = uv
            
            for vi, vv in zip(v_indices, v_vals):
                V[v_shift[vi], r] = vv
            
            for wi, wv in zip(w_indices, w_vals):
                W[w_shift[wi], r] = wv
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
