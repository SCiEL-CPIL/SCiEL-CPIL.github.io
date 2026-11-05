import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Total rank
    # 4x4x4 part: 49
    # 4x4x1 part: 16
    # 4x1x4 part: 16
    # 4x1x1 part: 4
    rank = 49 + 16 + 16 + 4
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to get 2x2x2 Strassen factors
    # Returns U, V, W of shape 4x7
    def get_strassen_2x2_factors():
        # U indices: 0:00, 1:01, 2:10, 3:11
        # V indices: 0:00, 1:01, 2:10, 3:11
        # W indices: 0:00, 1:01, 2:10, 3:11 (using k*n+i, n=2)
        
        u = [[0]*7 for _ in range(4)]
        v = [[0]*7 for _ in range(4)]
        w = [[0]*7 for _ in range(4)]
        
        # Term 0: (a00+a11)(b00+b11)
        # U: 1,0,0,1
        u[0][0]=1; u[3][0]=1
        # V: 1,0,0,1
        v[0][0]=1; v[3][0]=1
        # W: c00+1, c11+1 -> idx 0, 3
        w[0][0]=1; w[3][0]=1
        
        # Term 1: (a10+a11)b00
        # U: 0,0,1,1
        u[2][1]=1; u[3][1]=1
        # V: 1,0,0,0
        v[0][1]=1
        # W: c10+1, c11-1 -> idx 2, 3
        w[2][1]=1; w[3][1]=-1
        
        # Term 2: a00(b01-b11)
        # U: 1,0,0,0
        u[0][2]=1
        # V: 0,1,0,-1
        v[1][2]=1; v[3][2]=-1
        # W: c01+1, c11+1 -> idx 1, 3
        w[1][2]=1; w[3][2]=1
        
        # Term 3: a11(b10-b00)
        # U: 0,0,0,1
        u[3][3]=1
        # V: 0,0,1,-1
        v[2][3]=1; v[0][3]=-1
        # W: c00+1, c10+1 -> idx 0, 2
        w[0][3]=1; w[2][3]=1
        
        # Term 4: (a00+a01)b11
        # U: 1,1,0,0
        u[0][4]=1; u[1][4]=1
        # V: 0,0,0,1
        v[3][4]=1
        # W: c00-1, c01+1 -> idx 0, 1
        w[0][4]=-1; w[1][4]=1
        
        # Term 5: (a10-a00)(b00+b01)
        # U: -1,0,1,0
        u[0][5]=-1; u[2][5]=1
        # V: 1,1,0,0
        v[0][5]=1; v[1][5]=1
        # W: c11+1 -> idx 3
        w[3][5]=1
        
        # Term 6: (a01-a11)(b10+b11)
        # U: 0,1,0,-1
        u[1][6]=1; u[3][6]=-1
        # V: 0,0,1,1
        v[2][6]=1; v[3][6]=1
        # W: c00+1 -> idx 0
        w[0][6]=1
        
        return u, v, w

    s_u, s_v, s_w = get_strassen_2x2_factors()
    
    col = 0
    # 4x4x4 part
    # A blocks: br, bc in {0,1}
    # B blocks: br, bc in {0,1}
    # C blocks: br, bc in {0,1}
    
    # Strassen 4x4 terms:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # We represent each 4x4 term as a combination of 2x2 blocks.
    # For each 4x4 term, we have linear coeffs for A blocks and B blocks.
    # Then we multiply those linear combos using the 2x2x2 Strassen factors.
    
    # To make it simple, we can iterate through the 7 4x4 terms.
    # For each, we define the A-block coeffs and B-block coeffs.
    # Then for each of the 7 2x2 terms, we map to global indices.
    
    # A block (br, bc) corresponds to rows 2*br..2*br+1, cols 2*bc..2*bc+1 in 4x4.
    # B block (br, bc) corresponds to rows 2*br..2*br+1, cols 2*bc..2*bc+1 in 4x4.
    # C block (br, bc) corresponds to rows 2*br..2*br+1, cols 2*bc..2*bc+1 in 4x4.
    
    # 4x4 Term 1: (A11+A22)(B11+B22)
    # A coeffs: A11:1, A22:1. (A12:0, A21:0)
    # B coeffs: B11:1, B22:1. (B12:0, B21:0)
    # Result: C11+1, C22+1.
    
    # This approach of combining linear combos of blocks is tricky to implement generically
    # without expanding the 2x2 multiplication inside.
    
    # Instead, let's just list the 49 scalar multiplications explicitly for the 4x4x4 block.
    # It's tedious but safe.
    
    # Actually, we can construct the 4x4x4 factors by applying the 2x2x2 factors
    # to the "super-elements".
    
    # Let's stick to the block expansion.
    # For a given 4x4 term, say T1 = (A11+A22)(B11+B22).
    # This is a sum of products of 2x2 matrices.
    # (A11+B22) is a 2x2 matrix. Let's call it X.
    # (B11+B22) is a 2x2 matrix. Let's call it Y.
    # X*Y is a 2x2 matrix.
    # We can decompose X*Y using the 2x2 Strassen factors.
    # X = A11 + A22. So X's entries are sums of A11's and A22's entries.
    # Y = B11 + B22.
    
    # So for each of the 7 2x2 Strassen terms of X*Y:
    # We need to write the linear combination of A entries and B entries.
    
    # This is effectively doing the same thing as hardcoding the 49 terms.
    
    # Let's write a function to add a 2x2x2 decomposition into the global matrices.
    # It takes:
    # - U_coeffs: 4x4 matrix? No, just the linear combination of the 4 2x2 blocks.
    #   Actually, for a specific 2x2 term in the Strassen expansion of a 4x4 term,
    #   the U part is a linear combination of specific A entries.
    
    # Let's hardcode the 49 terms for 4x4x4. It's the most robust way.
    # I will generate them programmatically by composing the 2x2 factors with the 4x4 block structure.
    
    def add_2x2_term(U_mat, V_mat, W_mat, a_br, a_bc, b_br, b_bc, c_br, c_bc, coeff_u, coeff_v, coeff_w):
        # a_br, a_bc: block row/col of A (0 or 1)
        # b_br, b_bc: block row/col of B
        # c_br, c_bc: block row/col of C
        # coeff_u, coeff_v, coeff_w: coefficients for the 2x2 entries
        # This is getting complicated.
        
        pass

    # Alternative: Just use the 2x2x2 factors and map the indices.
    # For the 4x4x4 part, we have 7 "outer" terms.
    # For each outer term, we have 7 "inner" terms.
    
    # Outer Term 1: (A11+A22)(B11+B22)
    # A part: A11 (br=0,bc=0) coeff 1, A22 (br=1,bc=1) coeff 1.
    # B part: B11 (br=0,bc=0) coeff 1, B22 (br=1,bc=1) coeff 1.
    # C part: C11 (br=0,bc=0) coeff 1, C22 (br=1,bc=1) coeff 1.
    # Wait, the result of (A11+A22)(B11+B22) is a 2x2 matrix which is placed in...
    # No, A11 is 2x2. B11 is 2x2. Product is 2x2.
    # This 2x2 product contributes to C11?
    # In 4x4 multiplication:
    # C11 = A11B11 + A12B21
    # C12 = A11B12 + A12B22
    # C21 = A21B11 + A22B21
    # C22 = A21B12 + A22B22
    
    # Strassen M1 = (A11+A22)(B11+B22)
    # M1 contributes to C11 and C22.
    # C11 += M1 ...
    # C22 += M1 ...
    
    # So M1 is a 2x2 matrix.
    # We decompose M1 = X*Y where X=A11+A22, Y=B11+B22.
    # The 2x2 Strassen factors for X*Y give us 7 scalar products.
    # For each scalar product, we have:
    # u_vec (4 entries, coeffs of x00,x01,x10,x11)
    # v_vec (4 entries, coeffs of y00,y01,y10,y11)
    # w_vec (4 entries, coeffs of c00,c01,c10,c11 of the 2x2 result)
    
    # X = A11 + A22.
    # x00 = A11_00 + A22_00
    # x01 = A11_01 + A22_01
    # x10 = A11_10 + A22_10
    # x11 = A11_11 + A22_11
    
    # So for a given u_vec entry u_k for x_ii:
    # The global U coefficient for A11_ii is u_k * 1.
    # The global U coefficient for A22_ii is u_k * 1.
    
    # Similarly for V.
    
    # For W, the 2x2 result M1 is added to C11 and C22.
    # The 2x2 result indices are local (r,c) in {0,1}x{0,1}.
    # Global C11 index: i = 2*0+r, k=2*0+c?
    # No, C11 is the top-left 2x2 block of the 4x4 C.
    # Global i = 0..1, k = 0..1.
    # Global C22 is bottom-right 2x2 block.
    # Global i = 2..3, k = 2..3.
    
    # So for a w_vec entry w_l for local c_rr:
    # Global C11 (rr) gets w_l * 1.
    # Global C22 (rr) gets w_l * 1. (Wait, M1 contributes +1 to C11 and +1 to C22?
    # C11 = M1 + M4 - M5 + M7.
    # C22 = M1 - M2 + M3 + M6.
    # Yes, M1 has +1 in both.)
    
    # This structure allows us to program it.
    
    def add_4x4_term(a_block_coeffs, b_block_coeffs, c_block_coeffs, start_col):
        # a_block_coeffs: dict (br,bc) -> coeff
        # b_block_coeffs: dict (br,bc) -> coeff
        # c_block_coeffs: dict (br,bc) -> coeff
        # This term is sum of (sum a_b A_b) * (sum b_b B_b)
        # Let X = sum a_b A_b, Y = sum b_b B_b.
        # We decompose X*Y using 2x2 Strassen.
        
        for inner_k in range(7):
            # Get u_vec, v_vec, w_vec for this inner term
            u_vec = [s_u[r][inner_k] for r in range(4)] # 4 entries
            v_vec = [s_v[r][inner_k] for r in range(4)]
            w_vec = [s_w[r][inner_k] for r in range(4)]
            
            col_idx = start_col + inner_k
            # Wait, we need 49 columns.
            # This function adds 7 columns for one 4x4 term.
            
            # For each of the 4 local entries in u_vec (00, 01, 10, 11)
            # We need to distribute the coefficient to the global A entries.
            
            # u_vec[0] is coeff for x00.
            # x00 = sum a_b * A_b[0,0]
            # So global U row (i, j) corresponding to A_b[0,0] gets u_vec[0] * a_b.
            
            # Mapping local (r,c) to global (i,j) for a block (br,bc):
            # i = 2*br + r
            # j = 2*bc + c
            
            # We iterate over all A blocks in a_block_coeffs
            for (a_br, a_bc), a_coeff in a_block_coeffs.items():
                for lr in range(2):
                    for lc in range(2):
                        local_idx = lr * 2 + lc # 0:00, 1:01, 2:10, 3:11
                        u_val = u_vec[local_idx]
                        if u_val != 0:
                            global_i = 2 * a_br + lr
                            global_j = 2 * a_bc + lc
                            global_u_row = global_i * 5 + global_j
                            U[global_u_row][col_idx] += a_coeff * u_val
            
            # Similarly for V
            for (b_br, b_bc), b_coeff in b_block_coeffs.items():
                for lr in range(2):
                    for lc in range(2):
                        local_idx = lr * 2 + lc
                        v_val = v_vec[local_idx]
                        if v_val != 0:
                            global_j = 2 * b_br + lr
                            global_k = 2 * b_bc + lc
                            global_v_row = global_j * 5 + global_k
                            V[global_v_row][col_idx] += b_coeff * v_val
            
            # Similarly for W
            # w_vec[r] is coeff for local c_rr of the 2x2 product.
            # This 2x2 product is added to C blocks specified in c_block_coeffs.
            # If C block (c_br, c_bc) has coeff c_coeff, then local c_rr adds to global C at
            # i = 2*c_br + lr, k = 2*c_bc + lc.
            
            for (c_br, c_bc), c_coeff in c_block_coeffs.items():
                for lr in range(2):
                    for lc in range(2):
                        local_idx = lr * 2 + lc
                        w_val = w_vec[local_idx]
                        if w_val != 0:
                            global_k = 2 * c_bc + lc
                            global_i = 2 * c_br + lr
                            global_w_row = global_k * 4 + global_i
                            W[global_w_row][col_idx] += c_coeff * w_val

    # Define the 7 Strassen 4x4 terms
    # M1 = (A11+A22)(B11+B22)
    # A: {(0,0):1, (1,1):1}
    # B: {(0,0):1, (1,1):1}
    # C: {(0,0):1, (1,1):1}
    
    # M2 = (A21+A22)B11
    # A: {(1,0):1, (1,1):1}
    # B: {(0,0):1}
    # C: {(1,1):-1}  (C22 += -M2? No. C11=M1+M4-M5+M7. C22=M1-M2+M3+M6.)
    # Wait, M2 = (A21+A22)B11.
    # C10 = A21B11 + A22B21. No.
    # C21 = A21B11 + A22B21.
    # C22 = A21B12 + A22B22.
    # Strassen:
    # M2 = (A21+A22)B11.
    # C21 += M2. C22 += -M2.
    # So C: {(1,1):-1, (1,0):1}?
    # Let's re-verify Strassen 2x2 result placement.
    # M2 contributes to c10 (+1) and c11 (-1).
    # In 4x4, the 2x2 result of M2 is placed in the bottom-right?
    # No, M2 is a 2x2 matrix.
    # A21 is bottom-left of A. B11 is top-left of B.
    # Product is 2x2.
    # It contributes to C21 (bottom-left) and C22 (bottom-right)?
    # C21 = A21B11 + A22B21.
    # C22 = A21B12 + A22B22.
    # M2 = (A21+A22)B11 = A21B11 + A22B11.
    # So M2 has A21B11 (part of C21) and A22B11 (not in C21 or C22 directly?).
    # A22B11 is in C21? No.
    # C21 = A21B11 + A22B21.
    # So A22B11 is an extra term.
    # M6 = (A21-A11)(B11+B12) = A21B11 + A21B12 - A11B11 - A11B12.
    # C22 = M1 - M2 + M3 + M6.
    # M1 - M2 + M3 + M6 = ...
    # It works out.
    
    # So for M2, the 2x2 result contributes to C21 with +1 and C22 with -1.
    # C21 block is (1,0). C22 block is (1,1).
    # So C: {(1,0):1, (1,1):-1}.
    
    # M3 = A11(B12-B22)
    # A: {(0,0):1}
    # B: {(0,1):1, (1,1):-1}
    # C: C12 += M3, C22 += M3.
    # C12 block (0,1). C22 block (1,1).
    # C: {(0,1):1, (1,1):1}.
    
    # M4 = A22(B21-B11)
    # A: {(1,1):1}
    # B: {(1,0):1, (0,0):-1}
    # C: C11 += M4, C21 += M4.
    # C11 (0,0). C21 (1,0).
    # C: {(0,0):1, (1,0):1}.
    
    # M5 = (A11+A12)B22
    # A: {(0,0):1, (0,1):1}
    # B: {(1,1):1}
    # C: C11 -= M5, C12 += M5.
    # C11 (0,0). C12 (0,1).
    # C: {(0,0):-1, (0,1):1}.
    
    # M6 = (A21-A11)(B11+B12)
    # A: {(1,0):1, (0,0):-1}
    # B: {(0,0):1, (0,1):1}
    # C: C22 += M6.
    # C22 (1,1).
    # C: {(1,1):1}.
    
    # M7 = (A12-A22)(B21+B22)
    # A: {(0,1):1, (1,1):-1}
    # B: {(1,0):1, (1,1):1}
    # C: C11 += M7.
    # C11 (0,0).
    # C: {(0,0):1}.
    
    terms = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}), # M1
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}), # M2
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}), # M3
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}), # M4
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}), # M5
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}), # M6
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}), # M7
    ]
    
    for t_idx, (a_coeff, b_coeff, c_coeff) in enumerate(terms):
        start_col = t_idx * 7
        add_4x4_term(a_coeff, b_coeff, c_coeff, start_col)
        
    col = 49
    
    # 4x4x1 part: A_L (4x4) * B_LR (4x1) -> C (4x1)
    # Rank 16
    for i in range(4):
        for j in range(4):
            U[i*5 + j][col] = 1
            V[j*5 + 4][col] = 1 # j in 0..3, k=4
            W[4*4 + i][col] = 1 # k=4, i in 0..3
            col += 1
            
    # 4x1x4 part: A_R (4x1) * B_RL (1x4) -> C (4x4)
    # Rank 16
    for i in range(4):
        for k in range(4):
            U[i*5 + 4][col] = 1 # j=4
            V[4*5 + k][col] = 1 # j=4, k in 0..3
            W[k*4 + i][col] = 1
            col += 1
            
    # 4x1x1 part: A_R (4x1) * B_RR (1x1) -> C (4x1)
    # Rank 4
    for i in range(4):
        U[i*5 + 4][col] = 1
        V[4*5 + 4][col] = 1
        W[4*4 + i][col] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
