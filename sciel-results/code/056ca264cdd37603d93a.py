import numpy as np
import time

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank estimate: 2 * 49 (for 4x4x4 blocks) + 16 (for 4x4x1) = 114
    # We will build exactly this.
    
    # Pre-allocate U, V, W with max possible rank to avoid dynamic resizing issues in logic
    # Max rank 144 is safe.
    max_rank = 144
    U = [[0.0] * max_rank for _ in range(n * m)]
    V = [[0.0] * max_rank for _ in range(m * p)]
    W = [[0.0] * max_rank for _ in range(p * n)]
    
    current_rank = 0
    
    # Helper to add a rank-1 term
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        nonlocal current_rank
        col = current_rank
        U[u_idx][col] += u_val
        V[v_idx][col] += v_val
        W[w_idx][col] += w_val
        current_rank += 1

    # --- Strassen 2x2 Decomposition (Rank 7) ---
    # Indices for 2x2:
    # A: 0:00, 1:01, 2:10, 3:11 (i*2+j)
    # B: 0:00, 1:01, 2:10, 3:11 (j*2+k)
    # C: 0:00, 1:01, 2:10, 3:11 (k*2+i) -> Prompt says W[k*n+i]. For 2x2 block, local index is 2*k+i.
    
    # Strassen 2x2 terms:
    # M1: (A00+A11)(B00+B11)
    # M2: (A10+A11)B00
    # M3: A00(B10-B11)
    # M4: A11(B01-B00)
    # M5: (A00+A01)B11
    # M6: (A10-A00)(B00+B01)
    # M7: (A01-A11)(B10+B11)
    
    strassen_2x2_terms = [
        # (u_coeffs, v_coeffs, w_coeffs)
        # u: {idx: val}, v: {idx: val}, w: {idx: val}
        # M1
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 1: 0, 2: 0, 3: 1}), # C00+=1, C11+=1 (Wait, C00=M1+..., C11=M1-...)
        # Correction: W coefficients are for the output C.
        # C00 gets +1 from M1. C11 gets +1 from M1.
        # My W dict should map C_index to coefficient.
        # C indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1).
        # M1 contributes to C00 (idx 0) and C11 (idx 3).
        # So W = {0:1, 3:1}.
        
        # Let's re-verify Strassen 2x2 output combinations:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # M1: C00+=1, C11+=1
        # M2: C10+=1, C11+=-1
        # M3: C01+=1, C11+=1
        # M4: C00+=1, C10+=1
        # M5: C00+=-1, C01+=1
        # M6: C11+=1
        # M7: C00+=1
        
        # M1: U{0:1,3:1}, V{0:1,3:1}, W{0:1, 3:1}
        # M2: U{2:1,3:1}, V{0:1},     W{2:1, 3:-1}
        # M3: U{0:1},     V{2:1,3:-1},W{1:1, 3:1}
        # M4: U{3:1},     V{1:1,0:-1},W{0:1, 2:1}
        # M5: U{0:1,1:1}, V{3:1},     W{0:-1,1:1}
        # M6: U{2:1,0:-1},V{0:1,1:1}, W{3:1}
        # M7: U{1:1,3:-1},V{2:1,3:1}, W{0:1}
    ]
    
    strassen_2x2_data = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1},     {2: 1, 3: -1}),
        ({0: 1},     {2: 1, 3: -1},{1: 1, 3: 1}),
        ({3: 1},     {1: 1, 0: -1},{0: 1, 2: 1}),
        ({0: 1, 1: 1}, {3: 1},     {0: -1, 1: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1})
    ]

    # --- Strassen 4x4 Block Coefficients ---
    # A blocks: 00, 01, 10, 11
    # B blocks: 00, 01, 10, 11
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B10-B11)
    # P4 = A11(B01-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    
    # C blocks:
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # We need to map these to the global U, V, W.
    
    # Function to add a block multiplication
    # block_a: dict {block_idx: coeff}
    # block_b: dict {block_idx: coeff}
    # block_c: dict {block_idx: coeff}
    def add_block_mult(block_a, block_b, block_c):
        # For each combination of A-block and B-block, we have a scalar multiplication of 2x2 matrices.
        # We expand each 2x2 multiplication using Strassen 2x2 (rank 7).
        
        for a_blk, a_coeff in block_a.items():
            for b_blk, b_coeff in block_b.items():
                # The product is (a_coeff * A_a_blk) * (b_coeff * B_b_blk)
                # This is a 2x2 matrix multiplication.
                # We use the 7 Strassen terms for this 2x2 product.
                for term in strassen_2x2_data:
                    u_2x2, v_2x2, w_2x2 = term
                    
                    # Map 2x2 indices to global indices
                    # A_a_blk is at rows/cols determined by a_blk
                    # a_blk 0: (0,0) -> rows 0,1 cols 0,1
                    # a_blk 1: (0,1) -> rows 0,1 cols 2,3
                    # a_blk 2: (1,0) -> rows 2,3 cols 0,1
                    # a_blk 3: (1,1) -> rows 2,3 cols 2,3
                    
                    # Global U index: i*4 + j
                    # For a specific 2x2 element (local_i, local_j) in block a_blk:
                    # global_i = (a_blk // 2) * 2 + local_i
                    # global_j = (a_blk % 2) * 2 + local_j
                    # global_u_idx = global_i * 4 + global_j
                    
                    # Similarly for V (block b_blk):
                    # global_j = (b_blk // 2) * 2 + local_j
                    # global_k = (b_blk % 2) * 2 + local_k
                    # global_v_idx = global_j * 9 + global_k (Note: p=9, but B blocks are 4x4 part, so k is 0..3)
                    
                    # And for W (block c_blk):
                    # The output C block c_blk corresponds to global rows/cols.
                    # c_blk 0: (0,0) -> rows 0,1 cols 0,1 (of C1)
                    # c_blk 1: (0,1) -> rows 0,1 cols 2,3 (of C1)
                    # c_blk 2: (1,0) -> rows 2,3 cols 0,1 (of C1)
                    # c_blk 3: (1,1) -> rows 2,3 cols 2,3 (of C1)
                    
                    # However, block_c is distributed among P1..P7.
                    # We are processing one specific P_t.
                    # The coefficients in block_c apply to the output of this P_t.
                    # So for each local element in w_2x2, we add to the corresponding global C element.
                    
                    # Wait, the logic is:
                    # We are adding a term to the global decomposition.
                    # The term is: (a_coeff * L_u(A_a_blk)) * (b_coeff * L_v(B_b_blk)) * (w_coeff * E_c_local)
                    # This contributes to the global U, V, W.
                    
                    # Let's iterate over the non-zero entries of the 2x2 Strassen terms.
                    
                    # Scale the coefficients
                    scale = a_coeff * b_coeff
                    
                    for u_local, u_val in u_2x2.items():
                        # Map u_local to global A index
                        li = u_local // 2
                        lj = u_local % 2
                        gi = (a_blk // 2) * 2 + li
                        gj = (a_blk % 2) * 2 + lj
                        u_idx = gi * 4 + gj
                        
                        # Store U part
                        # We need to accumulate this into the final U matrix.
                        # Since we are inside a loop that adds a rank-1 component,
                        # we need to associate this U value with the current rank column.
                        
                        # This approach is hard because the 7 Strassen terms are separate rank-1 terms.
                        # So for each of the 7 terms, we call add_term.
                        
                        # But add_term needs a single u_idx, v_idx, w_idx.
                        # The Strassen term has multiple non-zeros in U, V, W.
                        # This is not a rank-1 term in the full space!
                        # It is a rank-1 term in the *block* space, but expanded to scalar space, it's a product of sums.
                        # (Sum A) * (Sum B) * (Sum C) is NOT rank 1. It's a product of linear forms.
                        # Wait. The decomposition is sum of rank-1 tensors.
                        # A rank-1 tensor is u * v * w.
                        # If u = u1 + u2, then u*v*w = (u1*u2)*v*w + ... No.
                        # (u1+u2) is a single vector in the space.
                        # So the term IS rank 1 in the full space.
                        # u_idx refers to the coordinate of the vector u.
                        # If u has multiple non-zeros, that's fine. The vector u is sparse.
                        # So add_term should handle the whole vector?
                        # No, my add_term adds a single value to a single index.
                        # I need to add the whole vector U, V, W for this rank-1 term.
                        
                        # Let's modify add_term to take dicts or lists.
                        pass

    # Revised add_term
    def add_term_full(u_dict, v_dict, w_dict):
        nonlocal current_rank
        col = current_rank
        for idx, val in u_dict.items():
            U[idx][col] += val
        for idx, val in v_dict.items():
            V[idx][col] += val
        for idx, val in w_dict.items():
            W[idx][col] += val
        current_rank += 1

    # Re-define add_block_mult using add_term_full
    def add_block_mult_full(block_a, block_b, block_c):
        for a_blk, a_coeff in block_a.items():
            for b_blk, b_coeff in block_b.items():
                # Calculate the 7 Strassen terms for this 2x2 product
                for term in strassen_2x2_data:
                    u_2x2, v_2x2, w_2x2 = term
                    
                    # Scale
                    scale = a_coeff * b_coeff
                    
                    # Construct global u_dict, v_dict, w_dict
                    u_global = {}
                    v_global = {}
                    w_global = {}
                    
                    for u_local, u_val in u_2x2.items():
                        li = u_local // 2
                        lj = u_local % 2
                        gi = (a_blk // 2) * 2 + li
                        gj = (a_blk % 2) * 2 + lj
                        idx = gi * 4 + gj
                        u_global[idx] = u_global.get(idx, 0) + u_val * scale
                    
                    for v_local, v_val in v_2x2.items():
                        # v_local corresponds to B matrix (j,k)
                        lj = v_local // 2
                        lk = v_local % 2
                        gj = (b_blk // 2) * 2 + lj
                        gk = (b_blk % 2) * 2 + lk
                        idx = gj * 9 + gk # p=9
                        v_global[idx] = v_global.get(idx, 0) + v_val * scale
                    
                    # w_2x2 corresponds to C matrix (k,i) locally in the block.
                    # The block_c dict tells us which global C block this output goes to, and with what coefficient.
                    # BUT: block_c is the same for all a_blk, b_blk?
                    # No, block_c is defined per P_t.
                    # In the loop for P_t, block_a and block_b are fixed for that P_t.
                    # The output of this 2x2 multiplication goes to C block c_blk with coeff c_coeff.
                    # Wait, P_t is a product of (Linear A) and (Linear B).
                    # The result is a 2x2 matrix.
                    # This 2x2 matrix is added to specific C blocks.
                    # So for each c_blk in block_c, we add the output.
                    
                    for c_blk, c_coeff in block_c.items():
                        for w_local, w_val in w_2x2.items():
                            # w_local is index in 2x2 C block (k,i)
                            lk = w_local // 2
                            li = w_local % 2
                            gk = (c_blk // 2) * 2 + lk
                            gi = (c_blk % 2) * 2 + li
                            idx = gk * 4 + gi # W index: k*n + i
                            w_global[idx] = w_global.get(idx, 0) + w_val * c_coeff * scale
                            
                    # If w_global is empty, skip? No, it should have entries.
                    add_term_full(u_global, v_global, w_global)

    # Define the 7 P terms for 4x4
    # P1
    add_block_mult_full(
        {0: 1, 3: 1}, # A00+A11
        {0: 1, 3: 1}, # B00+B11
        {0: 1, 3: 1}  # C00+1, C11+1
    )
    # P2
    add_block_mult_full(
        {2: 1, 3: 1}, # A10+A11
        {0: 1},       # B00
        {2: 1, 3: -1} # C10+1, C11-1
    )
    # P3
    add_block_mult_full(
        {0: 1},       # A00
        {2: 1, 3: -1},# B10-B11
        {1: 1, 3: 1}  # C01+1, C11+1
    )
    # P4
    add_block_mult_full(
        {3: 1},       # A11
        {1: 1, 0: -1},# B01-B00
        {0: 1, 2: 1}  # C00+1, C10+1
    )
    # P5
    add_block_mult_full(
        {0: 1, 1: 1}, # A00+A01
        {3: 1},       # B11
        {0: -1, 1: 1} # C00-1, C01+1
    )
    # P6
    add_block_mult_full(
        {2: 1, 0: -1},# A10-A00
        {0: 1, 1: 1}, # B00+B01
        {3: 1}        # C11+1
    )
    # P7
    add_block_mult_full(
        {1: 1, 3: -1},# A01-A11
        {2: 1, 3: 1}, # B10+B11
        {0: 1}        # C00+1
    )
    
    # This handled the first 4x4x4 (C1, cols 0-3 of B).
    # We need to do it again for C2 (cols 4-7 of B).
    # The A part is the same. The B part is shifted.
    # We can't just reuse the function easily because V indices are different.
    # Let's modify add_block_mult_full to take a V_offset.
    
    # Reset current_rank? No, we are adding more terms.
    # Let's create a wrapper that handles the offset.
    
    def add_block_mult_full_offset(block_a, block_b, block_c, v_col_offset):
        for a_blk, a_coeff in block_a.items():
            for b_blk, b_coeff in block_b.items():
                for term in strassen_2x2_data:
                    u_2x2, v_2x2, w_2x2 = term
                    scale = a_coeff * b_coeff
                    u_global = {}
                    v_global = {}
                    w_global = {}
                    for u_local, u_val in u_2x2.items():
                        li = u_local // 2
                        lj = u_local % 2
                        gi = (a_blk // 2) * 2 + li
                        gj = (a_blk % 2) * 2 + lj
                        idx = gi * 4 + gj
                        u_global[idx] = u_global.get(idx, 0) + u_val * scale
                    for v_local, v_val in v_2x2.items():
                        lj = v_local // 2
                        lk = v_local % 2
                        gj = (b_blk // 2) * 2 + lj
                        gk = (b_blk % 2) * 2 + lk
                        idx = gj * 9 + (gk + v_col_offset)
                        v_global[idx] = v_global.get(idx, 0) + v_val * scale
                    for c_blk, c_coeff in block_c.items():
                        for w_local, w_val in w_2x2.items():
                            lk = w_local // 2
                            li = w_local % 2
                            gk = (c_blk // 2) * 2 + lk
                            gi = (c_blk % 2) * 2 + li
                            idx = gk * 4 + gi
                            w_global[idx] = w_global.get(idx, 0) + w_val * c_coeff * scale
                    add_term_full(u_global, v_global, w_global)

    # Add second 4x4x4 block (cols 4-7)
    add_block_mult_full_offset({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}, 4)
    add_block_mult_full_offset({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}, 4)
    add_block_mult_full_offset({0: 1}, {2: 1, 3: -1}, {1: 1, 3: 1}, 4)
    add_block_mult_full_offset({3: 1}, {1: 1, 0: -1}, {0: 1, 2: 1}, 4)
    add_block_mult_full_offset({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}, 4)
    add_block_mult_full_offset({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}, 4)
    add_block_mult_full_offset({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}, 4)

    # Add schoolbook for 4x4x1 (col 8)
    for i in range(4):
        for j in range(4):
            u_idx = i * 4 + j
            v_idx = j * 9 + 8
            w_idx = 8 * 4 + i
            add_term_full({u_idx: 1}, {v_idx: 1}, {w_idx: 1})

    # Trim matrices to current_rank
    U = [row[:current_rank] for row in U]
    V = [row[:current_rank] for row in V]
    W = [row[:current_rank] for row in W]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
