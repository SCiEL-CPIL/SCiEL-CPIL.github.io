def construct():
    n, m, p = 4, 4, 7
    
    # We will build the decomposition in two parts:
    # Part 1: 4x4x4 multiplication (first 4 columns of B)
    # Part 2: 4x4x3 multiplication (last 3 columns of B)
    
    # Helper to get global indices
    def a_idx(i, j): return i * m + j
    def b_idx(j, k): return j * p + k
    def c_idx(k, i): return k * n + i
    
    U_list = []
    V_list = []
    W_list = []
    
    # --- Part 1: 4x4x4 ---
    # Strassen blocks for A and B (first 4 cols)
    # A blocks: A00, A01, A10, A11 (2x2 each)
    # B blocks: B00, B01, B10, B11 (2x2 each)
    
    # Strassen terms definitions
    # Each term is (u_coeffs, v_coeffs, w_coeffs) for 2x2x2
    # u_coeffs: dict {local_a_idx: coeff}
    # v_coeffs: dict {local_b_idx: coeff}
    # w_coeffs: dict {local_c_idx: coeff}
    
    # Local indices for 2x2: 0,1,2,3
    # 0:00, 1:01, 2:10, 3:11
    
    # Standard Strassen rank-7 decomposition for 2x2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We need to expand each M into rank-1 terms?
    # No, M itself is a product of two linear forms.
    # So each M is ONE rank-1 term in the 2x2x2 tensor?
    # Yes. The rank-7 decomposition of 2x2x2 consists of these 7 terms.
    
    # So for each Strassen multiplication, we add 7 terms.
    
    # Let's define the 7 terms for a generic 2x2 multiplication
    # Term 1:
    u1 = {0: 1, 3: 1}
    v1 = {0: 1, 3: 1}
    w1 = {0: 1, 3: 1} # C00, C11
    
    # Wait, the W vector for a rank-1 term in 2x2x2 is the coefficient of C.
    # C00 gets +1, C11 gets +1.
    # So w1 = {0: 1, 3: 1} is correct.
    
    # Term 2:
    u2 = {2: 1, 3: 1}
    v2 = {0: 1}
    w2 = {2: 1, 3: -1} # C10 gets +1, C11 gets -1
    
    # Term 3:
    u3 = {0: 1}
    v3 = {1: 1, 3: -1}
    w3 = {1: 1} # C01 gets +1
    
    # Term 4:
    u4 = {3: 1}
    v4 = {2: 1, 0: -1}
    w4 = {0: 1, 2: 1} # C00 gets +1, C10 gets +1
    
    # Term 5:
    u5 = {0: 1, 1: 1}
    v5 = {3: 1}
    w5 = {0: -1, 1: 1} # C00 gets -1, C01 gets +1
    
    # Term 6:
    u6 = {2: 1, 0: -1}
    v6 = {0: 1, 1: 1}
    w6 = {3: 1} # C11 gets +1
    
    # Term 7:
    u7 = {1: 1, 3: -1}
    v7 = {2: 1, 3: 1}
    w7 = {0: 1, 3: 1} # C00 gets +1, C11 gets +1
    
    # Let's verify C11: M1 - M2 + M3 + M6
    # M1: +1
    # M2: -1
    # M3: 0
    # M6: +1
    # Sum: 1 - 1 + 0 + 1 = 1. Correct.
    
    # Let's verify C00: M1 + M4 - M5 + M7
    # M1: +1
    # M4: +1
    # M5: -1
    # M7: +1
    # Sum: 1 + 1 - 1 + 1 = 2? 
    # Wait. Strassen formula for C00 is M1 + M4 - M5 + M7.
    # Let's re-verify Strassen.
    # C00 = M1 + M4 - M5 + M7
    # M1 = (A00+A11)(B00+B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M7 = (A01-A11)(B10+B11)
    
    # Let's check coefficient of A00 B00 in C00.
    # M1: A00 B00 (coeff 1)
    # M4: 0
    # M5: 0
    # M7: 0
    # Sum: 1. Correct.
    
    # Coefficient of A00 B11 in C00.
    # M1: A00 B11 (coeff 1)
    # M4: 0
    # M5: A00 B11 (coeff -1)
    # M7: 0
    # Sum: 0. Correct.
    
    # Coefficient of A11 B00 in C00.
    # M1: A11 B00 (coeff 1)
    # M4: -A11 B00 (coeff -1)
    # M5: 0
    # M7: 0
    # Sum: 0. Correct.
    
    # Coefficient of A11 B11 in C00.
    # M1: A11 B11 (coeff 1)
    # M4: 0
    # M5: 0
    # M7: -A11 B11 (coeff -1)
    # Sum: 0. Correct.
    
    # So the W vectors are correct.
    
    strassen_terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        ({0: 1}, {1: 1, 3: -1}, {1: 1}),
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1, 3: 1})
    ]
    
    # Function to add a block multiplication to the lists
    # block_a_offsets: list of 4 global indices for A block
    # block_b_offsets: list of 4 global indices for B block
    # block_c_offsets: list of 4 global indices for C block
    def add_block_mult(a_off, b_off, c_off):
        for u_loc, v_loc, w_loc in strassen_terms:
            u_vec = [0] * (n*m)
            v_vec = [0] * (m*p)
            w_vec = [0] * (p*n)
            
            for loc, coeff in u_loc.items():
                u_vec[a_off[loc]] = coeff
            for loc, coeff in v_loc.items():
                v_vec[b_off[loc]] = coeff
            for loc, coeff in w_loc.items():
                w_vec[c_off[loc]] = coeff
                
            U_list.append(u_vec)
            V_list.append(v_vec)
            W_list.append(w_vec)
            
    # Define offsets for A blocks in 4x4
    # A00: 0,1,4,5
    # A01: 2,3,6,7
    # A10: 8,9,12,13
    # A11: 10,11,14,15
    a00 = [a_idx(0,0), a_idx(0,1), a_idx(1,0), a_idx(1,1)]
    a01 = [a_idx(0,2), a_idx(0,3), a_idx(1,2), a_idx(1,3)]
    a10 = [a_idx(2,0), a_idx(2,1), a_idx(3,0), a_idx(3,1)]
    a11 = [a_idx(2,2), a_idx(2,3), a_idx(3,2), a_idx(3,3)]
    
    # Define offsets for B blocks in 4x4 (first 4 cols)
    # B00: 0,1,4,5 (j=0,1; k=0,1)
    # B01: 2,3,6,7 (j=0,1; k=2,3)
    # B10: 8,9,12,13 (j=2,3; k=0,1)
    # B11: 10,11,14,15 (j=2,3; k=2,3)
    b00 = [b_idx(0,0), b_idx(0,1), b_idx(1,0), b_idx(1,1)]
    b01 = [b_idx(0,2), b_idx(0,3), b_idx(1,2), b_idx(1,3)]
    b10 = [b_idx(2,0), b_idx(2,1), b_idx(3,0), b_idx(3,1)]
    b11 = [b_idx(2,2), b_idx(2,3), b_idx(3,2), b_idx(3,3)]
    
    # Define offsets for C blocks in 4x4
    # C00: k=0,1; i=0,1 -> 0,4,1,5?
    # c_idx(k, i) = k*n + i
    # C00 (k=0,1; i=0,1):
    # k=0: i=0,1 -> 0,1
    # k=1: i=0,1 -> 4,5
    # Order in local 2x2: 00, 01, 10, 11
    # Local 0: k=0, i=0 -> 0
    # Local 1: k=0, i=1 -> 1
    # Local 2: k=1, i=0 -> 4
    # Local 3: k=1, i=1 -> 5
    c00 = [c_idx(0,0), c_idx(0,1), c_idx(1,0), c_idx(1,1)]
    
    # C01 (k=0,1; i=2,3)
    # Local 0: k=0, i=2 -> 2
    # Local 1: k=0, i=3 -> 3
    # Local 2: k=1, i=2 -> 6
    # Local 3: k=1, i=3 -> 7
    c01 = [c_idx(0,2), c_idx(0,3), c_idx(1,2), c_idx(1,3)]
    
    # C10 (k=2,3; i=0,1)
    # Local 0: k=2, i=0 -> 8
    # Local 1: k=2, i=1 -> 9
    # Local 2: k=3, i=0 -> 12
    # Local 3: k=3, i=1 -> 13
    c10 = [c_idx(2,0), c_idx(2,1), c_idx(3,0), c_idx(3,1)]
    
    # C11 (k=2,3; i=2,3)
    # Local 0: k=2, i=2 -> 10
    # Local 1: k=2, i=3 -> 11
    # Local 2: k=3, i=2 -> 14
    # Local 3: k=3, i=3 -> 15
    c11 = [c_idx(2,2), c_idx(2,3), c_idx(3,2), c_idx(3,3)]
    
    # Strassen combinations:
    # M1: A00+A11, B00+B11 -> C00, C11
    # We can't just call add_block_mult directly because the inputs are sums.
    # We need to modify the u_loc and v_loc dictionaries to reflect the global sums.
    
    # Let's rewrite the loop to handle the specific Strassen terms.
    
    # M1: (A00+A11)(B00+B11)
    u1 = {0: 1, 1: 1, 2: 1, 3: 1} # All A00 and A11
    # Map local A00 indices to global: a00
    # Map local A11 indices to global: a11
    # We need to merge these.
    
    # It's easier to just construct the vectors directly.
    
    def add_strassen_term(u_a_blocks, v_b_blocks, w_c_blocks):
        # u_a_blocks: list of (block_offset_list, multiplier)
        # v_b_blocks: list of (block_offset_list, multiplier)
        # w_c_blocks: list of (block_offset_list, multiplier)
        
        u_vec = [0] * (n*m)
        v_vec = [0] * (m*p)
        w_vec = [0] * (p*n)
        
        for off, mult in u_a_blocks:
            for idx in off:
                u_vec[idx] += mult
        for off, mult in v_b_blocks:
            for idx in off:
                v_vec[idx] += mult
        for off, mult in w_c_blocks:
            for idx in off:
                w_vec[idx] += mult
                
        U_list.append(u_vec)
        V_list.append(v_vec)
        W_list.append(w_vec)

    # M1
    add_strassen_term(
        [(a00, 1), (a11, 1)],
        [(b00, 1), (b11, 1)],
        [(c00, 1), (c11, 1)]
    )
    
    # M2
    add_strassen_term(
        [(a10, 1), (a11, 1)],
        [(b00, 1)],
        [(c10, 1), (c11, -1)]
    )
    
    # M3
    add_strassen_term(
        [(a00, 1)],
        [(b01, 1), (b11, -1)],
        [(c01, 1)]
    )
    
    # M4
    add_strassen_term(
        [(a11, 1)],
        [(b10, 1), (b00, -1)],
        [(c00, 1), (c10, 1)]
    )
    
    # M5
    add_strassen_term(
        [(a00, 1), (a01, 1)],
        [(b11, 1)],
        [(c00, -1), (c01, 1)]
    )
    
    # M6
    add_strassen_term(
        [(a10, 1), (a00, -1)],
        [(b00, 1), (b01, 1)],
        [(c11, 1)]
    )
    
    # M7
    add_strassen_term(
        [(a01, 1), (a11, -1)],
        [(b10, 1), (b11, 1)],
        [(c00, 1), (c11, 1)]
    )
    
    # Part 1 done. Rank 7.
    
    # --- Part 2: 4x4x3 ---
    # Split B into B_top (2x3) and B_bottom (2x3)?
    # No, we need to multiply A (4x4) by B2 (4x3).
    # Split A into 2x2 blocks.
    # Split B2 into 2x3 blocks.
    # C2 = A00 B2_0 + A01 B2_1 + A10 B2_0 + A11 B2_1
    # Each term is 2x2 x 3.
    # Rank of 2x2x3 is 11 (using 2x2x2 + 2x2x1).
    
    # We need to implement the rank-11 decomposition for 2x2x3.
    # Let's define it.
    # B2 is 2x3. Split into B2_2 (2x2) and B2_1 (2x1).
    # C2 = A B2_2 + A B2_1.
    # Term 1: A B2_2 (Rank 7)
    # Term 2: A B2_1 (Rank 4)
    
    # We need to map the local indices.
    # A block: 2x2.
    # B2 block: 2x3.
    # C2 block: 2x3.
    
    # Let's define the offsets for the 4x4x3 part.
    # A blocks are the same as before: a00, a01, a10, a11.
    # B2 blocks:
    # B2_0 (rows 0,1; cols 4,5,6):
    # j=0: k=4,5,6 -> b_idx(0,4), b_idx(0,5), b_idx(0,6)
    # j=1: k=4,5,6 -> b_idx(1,4), b_idx(1,5), b_idx(1,6)
    # B2_1 (rows 2,3; cols 4,5,6):
    # j=2: k=4,5,6 -> b_idx(2,4), b_idx(2,5), b_idx(2,6)
    # j=3: k=4,5,6 -> b_idx(3,4), b_idx(3,5), b_idx(3,6)
    
    # C2 blocks:
    # C2_0 (rows 0,1; cols 4,5,6):
    # k=4,5,6; i=0,1
    # C2_1 (rows 2,3; cols 4,5,6):
    # k=4,5,6; i=2,3
    
    # We need a function to add a 2x2x3 multiplication.
    # Inputs: A_off (4), B_off (6), C_off (6).
    # B_off is split into B_off_2 (4) and B_off_1 (2).
    # C_off is split into C_off_2 (4) and C_off_1 (2).
    
    def add_2x2x3(a_off, b_off, c_off):
        # b_off: [b00, b01, b02, b10, b11, b12]
        # c_off: [c00, c01, c02, c10, c11, c12]
        
        # Split B into 2x2 and 2x1
        # B2: b00, b01, b10, b11
        # B1: b02, b12
        b2 = [b_off[0], b_off[1], b_off[3], b_off[4]]
        b1 = [b_off[2], b_off[5]]
        
        # Split C into 2x2 and 2x1
        c2 = [c_off[0], c_off[1], c_off[3], c_off[4]]
        c1 = [c_off[2], c_off[5]]
        
        # Add 2x2x2 part (Rank 7)
        # We can reuse the strassen terms logic, but with different offsets.
        # The strassen terms are defined for local 2x2 indices.
        # We need to map local 2x2 indices to the specific b2/c2 offsets.
        
        # Let's just hardcode the 7 Strassen terms for this block.
        # M1: (A00+A11)(B00+B11)
        add_strassen_term(
            [(a_off[0:4], 1)], # This is wrong. a_off is flat.
            [],
            []
        )
        
        # Let's refactor add_strassen_term to take flat lists and multipliers.
        # It's getting complicated.
        
        # Alternative: Just use schoolbook for 2x2x3? Rank 12.
        # Total rank for 4x4x3 would be 4 * 12 = 48.
        # Total rank 7 + 48 = 55? No, 49 + 48 = 97.
        # 97 is still better than 112.
        # Let's stick to schoolbook for the 4x4x3 part to ensure correctness and simplicity.
        # Rank 48 for 4x4x3.
        
    # Let's implement schoolbook for 4x4x3.
    # For i in 0..3, j in 0..3, k in 4..6
    # U[i*m+j] = 1
    # V[j*p+k] = 1
    # W[k*n+i] = 1
    
    for i in range(4):
        for j in range(4):
            for k in range(4, 7):
                u_vec = [0] * (n*m)
                v_vec = [0] * (m*p)
                w_vec = [0] * (p*n)
                
                u_vec[i*m + j] = 1
                v_vec[j*p + k] = 1
                w_vec[k*n + i] = 1
                
                U_list.append(u_vec)
                V_list.append(v_vec)
                W_list.append(w_vec)
                
    # Convert lists to numpy arrays?
    # The task asks for lists of lists? "factor matrices of shapes..."
    # The example returns lists.
    
    return {"U": U_list, "V": V_list, "W": W_list}
