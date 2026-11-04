import numpy as np

def construct():
    n, m, p = 4, 4, 9
    
    # We will build U, V, W lists of lists
    U = []
    V = []
    W = []
    
    # Helper to add a term
    # u: list of length n*m
    # v: list of length m*p
    # w: list of length p*n
    def add_term(u, v, w):
        U.append(u)
        V.append(v)
        W.append(w)
        
    # Strassen 2x2 decomposition
    # Returns list of (u_2x2, v_2x2, w_2x2) where u, v, w are 4x1 vectors (flattened)
    # u corresponds to X, v to Y, w to C
    # X, Y are 2x2. C = XY.
    # Indices for 2x2: 0,1,2,3 -> (0,0), (0,1), (1,0), (1,1)
    
    def strassen_2x2():
        terms = []
        # M1 = (X00+X11)(Y00+Y11)
        # M2 = (X10+X11)Y00
        # M3 = X00(Y01-Y11)
        # M4 = X11(Y10-Y00)
        # M5 = (X00+X01)Y11
        # M6 = (X10-X00)(Y00+Y01)
        # M7 = (X01-X11)(Y10+Y11)
        
        # We need to map these to the full 4x4 indices later.
        # For now, just return the coefficients for X, Y, C
        # X indices: 0,1,2,3
        # Y indices: 0,1,2,3
        # C indices: 0,1,2,3
        
        # M1
        u1 = [1, 0, 0, 1]
        v1 = [1, 0, 0, 1]
        w1 = [1, 0, 0, 1] # C00 += M1, C01 += M1, C10 += M1, C11 += M1? 
        # No, Strassen formulas:
        # C00 = M1 - M4 + M6 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M3 + M5 + M7
        
        # So we add 7 terms, each contributing to specific C entries.
        
        # Term 1: M1
        # u: [1,0,0,1]
        # v: [1,0,0,1]
        # w: [1, 0, 0, 1] -> C00 gets +1, C11 gets +1?
        # Wait, M1 contributes to C00 and C11.
        # C00 = ... + M1 ...
        # C11 = ... + M1 ...
        # So w should have 1 at C00 and 1 at C11.
        w1 = [1, 0, 0, 1]
        terms.append((u1, v1, w1))
        
        # Term 2: M2
        u2 = [0, 0, 1, 1]
        v2 = [1, 0, 0, 0]
        w2 = [0, 0, 1, 0] # C10 gets +M2
        terms.append((u2, v2, w2))
        
        # Term 3: M3
        u3 = [1, 0, 0, 0]
        v3 = [0, 1, 0, -1]
        w3 = [0, 1, 0, -1] # C01 gets +M3, C11 gets -M3
        terms.append((u3, v3, w3))
        
        # Term 4: M4
        u4 = [0, 0, 0, 1]
        v4 = [0, 0, 1, -1]
        w4 = [-1, 0, 1, 0] # C00 gets -M4, C10 gets +M4
        terms.append((u4, v4, w4))
        
        # Term 5: M5
        u5 = [1, 1, 0, 0]
        v5 = [0, 0, 0, 1]
        w5 = [0, 1, 0, 1] # C01 gets +M5, C11 gets +M5
        terms.append((u5, v5, w5))
        
        # Term 6: M6
        u6 = [0, 0, 1, -1]
        v6 = [1, 1, 0, 0]
        w6 = [1, 0, 0, 0] # C00 gets +M6
        terms.append((u6, v6, w6))
        
        # Term 7: M7
        u7 = [0, 1, 0, -1]
        v7 = [0, 0, 1, 1]
        w7 = [0, 0, 0, 1] # C11 gets +M7
        terms.append((u7, v7, w7))
        
        return terms

    strassen_terms = strassen_2x2()
    
    # Now, for 4x4 multiplication.
    # A is 4x4. Blocks A11, A12, A21, A22 are 2x2.
    # B is 4x4. Blocks B11, B12, B21, B22 are 2x2.
    
    # We need to map the 2x2 Strassen terms to the 4x4 indices.
    # For each of the 7 "block products" P_k, we use the 7 Strassen terms.
    # P_k = X_k Y_k.
    # X_k and Y_k are linear combinations of the blocks of A and B.
    # Each block is 2x2.
    # So X_k is a 2x2 matrix of 2x2 blocks? No.
    # X_k is a 2x2 matrix formed by adding blocks.
    # Example: X_1 = A11 + A22.
    # A11 is top-left 2x2 of A.
    # A22 is bottom-right 2x2 of A.
    # So X_1 is a 2x2 matrix.
    # We apply Strassen 2x2 to X_1 and Y_1.
    # The entries of X_1 are linear combinations of entries of A.
    
    # Let's define the block additions for the 7 P_k.
    # P1: X=A11+A22, Y=B11+B22
    # P2: X=A21+A22, Y=B11
    # P3: X=A11, Y=B12-B22
    # P4: X=A22, Y=B21-B11
    # P5: X=A11+A12, Y=B22
    # P6: X=A21-A11, Y=B11+B12
    # P7: X=A12-A22, Y=B21+B22
    
    # For each P_k, we have X_k and Y_k.
    # We iterate over the 7 Strassen terms (u_t, v_t, w_t) for 2x2.
    # u_t is a vector of length 4, representing coefficients for X_k's entries.
    # X_k's entries are themselves linear combinations of A's entries.
    
    # We need to compose these linear combinations.
    
    # Let's create a function to expand a 2x2 block combination into a 16-vector for U.
    # Block indices: 0=A11, 1=A12, 2=A21, 3=A22.
    # Each block is 2x2, so 4 entries.
    # A11 entries: 0,1,4,5 (row-major 4x4: 0,1,2,3, 4,5,6,7...)
    # Wait, A is 4x4.
    # A11: rows 0-1, cols 0-1. Indices: 0,1,4,5.
    # A12: rows 0-1, cols 2-3. Indices: 2,3,6,7.
    # A21: rows 2-3, cols 0-1. Indices: 8,9,12,13.
    # A22: rows 2-3, cols 2-3. Indices: 10,11,14,15.
    
    def get_block_indices(block_id):
        if block_id == 0: # A11
            return [0, 1, 4, 5]
        elif block_id == 1: # A12
            return [2, 3, 6, 7]
        elif block_id == 2: # A21
            return [8, 9, 12, 13]
        elif block_id == 3: # A22
            return [10, 11, 14, 15]
            
    def expand_u(coeff_2x2, block_coeffs):
        # coeff_2x2: list of 4 floats, coefficients for the 4 entries of the 2x2 matrix X_k
        # block_coeffs: list of 4 floats, coefficients for the 4 blocks A11..A22
        # Returns list of 16 floats
        u_full = [0.0] * 16
        # X_k = sum_{b=0}^3 block_coeffs[b] * A_b
        # Entry (i,j) of X_k is sum_{b} block_coeffs[b] * A_b[i,j]
        # A_b[i,j] is at index idx in A.
        # So u_full[idx] += coeff_2x2[i*2+j] * block_coeffs[b]
        
        for i in range(2):
            for j in range(2):
                c = coeff_2x2[i*2+j]
                if c == 0: continue
                for b in range(4):
                    bc = block_coeffs[b]
                    if bc == 0: continue
                    idxs = get_block_indices(b)
                    idx = idxs[i*2+j]
                    u_full[idx] += c * bc
        return u_full

    # Similarly for V and W.
    # B blocks: B11, B12, B21, B22.
    # B is 4x4.
    # B11: 0,1,4,5
    # B12: 2,3,6,7
    # B21: 8,9,12,13
    # B22: 10,11,14,15
    
    def get_block_indices_b(block_id):
        return get_block_indices(block_id) # Same structure
        
    def expand_v(coeff_2x2, block_coeffs):
        v_full = [0.0] * 16
        for i in range(2):
            for j in range(2):
                c = coeff_2x2[i*2+j]
                if c == 0: continue
                for b in range(4):
                    bc = block_coeffs[b]
                    if bc == 0: continue
                    idxs = get_block_indices_b(b)
                    idx = idxs[i*2+j]
                    v_full[idx] += c * bc
        return v_full

    # W blocks: C11, C12, C21, C22.
    # C is 4x4.
    # W shape is (16, rank).
    # C11: 0,1,4,5
    # C12: 2,3,6,7
    # C21: 8,9,12,13
    # C22: 10,11,14,15
    
    def expand_w(coeff_2x2, block_coeffs):
        w_full = [0.0] * 16
        for i in range(2):
            for j in range(2):
                c = coeff_2x2[i*2+j]
                if c == 0: continue
                for b in range(4):
                    bc = block_coeffs[b]
                    if bc == 0: continue
                    idxs = get_block_indices(b)
                    idx = idxs[i*2+j]
                    w_full[idx] += c * bc
        return w_full

    # Define the 7 block products for 4x4 Strassen
    # X_blocks, Y_blocks
    # P1: X=A11+A22 -> [1,0,0,1], Y=B11+B22 -> [1,0,0,1]
    # P2: X=A21+A22 -> [0,0,1,1], Y=B11 -> [1,0,0,0]
    # P3: X=A11 -> [1,0,0,0], Y=B12-B22 -> [0,1,0,-1]
    # P4: X=A22 -> [0,0,0,1], Y=B21-B11 -> [0,-1,1,0]
    # P5: X=A11+A12 -> [1,1,0,0], Y=B22 -> [0,0,0,1]
    # P6: X=A21-A11 -> [-1,0,1,0], Y=B11+B12 -> [1,1,0,0]
    # P7: X=A12-A22 -> [0,1,0,-1], Y=B21+B22 -> [0,0,1,1]
    
    block_products = [
        ([1,0,0,1], [1,0,0,1]),
        ([0,0,1,1], [1,0,0,0]),
        ([1,0,0,0], [0,1,0,-1]),
        ([0,0,0,1], [0,-1,1,0]),
        ([1,1,0,0], [0,0,0,1]),
        ([-1,0,1,0], [1,1,0,0]),
        ([0,1,0,-1], [0,0,1,1])
    ]
    
    # Generate terms for B1 (4x4)
    for x_blocks, y_blocks in block_products:
        for u_t, v_t, w_t in strassen_terms:
            u_full = expand_u(u_t, x_blocks)
            v_full = expand_v(v_t, y_blocks)
            w_full = expand_w(w_t, [1,1,1,1]) # Wait, w_t is for C blocks?
            # No, w_t is the coefficient for the 2x2 result of X_k Y_k.
            # The result P_k is a 2x2 matrix.
            # P_k contributes to C blocks.
            # P1 contributes to C11, C12, C21, C22?
            # No, P_k is a single 2x2 matrix.
            # In Strassen 4x4, the 7 products P_k are 2x2 matrices.
            # C = sum of linear combinations of P_k.
            # C11 = P1 - P4 + P6 - P7
            # C12 = P3 + P5
            # C21 = P2 + P4
            # C22 = P1 - P3 + P5 + P7
            
            # So we need to map the 2x2 result of each P_k to the 4x4 C.
            # The 2x2 result of P_k is stored in w_t (coefficients for the 4 entries of the 2x2 result).
            # But w_t from strassen_2x2() gives the contribution to the 2x2 C.
            # We need to distribute this 2x2 C into the 4x4 C blocks.
            
            # Let's re-evaluate.
            # The 7 terms from strassen_2x2() decompose the multiplication of two 2x2 matrices X and Y into 7 rank-1 tensors.
            # Each tensor is u_t (coeffs for X), v_t (coeffs for Y), w_t (coeffs for C).
            # Here X and Y are the block combinations for P_k.
            # The result C (2x2) is P_k.
            # So w_t gives the coefficients for the entries of P_k.
            # P_k is a 2x2 matrix.
            # We need to add P_k to the final 4x4 C with appropriate signs.
            
            # So for each P_k, we have a "sign vector" for the 4 blocks of C.
            # P1: C11 += 1, C22 += 1? No.
            # C11 = P1 - P4 + P6 - P7
            # C12 = P3 + P5
            # C21 = P2 + P4
            # C22 = P1 - P3 + P5 + P7
            
            # So for P1, it contributes +1 to C11 and +1 to C22?
            # No, P1 is a 2x2 matrix.
            # C11 is a 2x2 block.
            # So P1 is added to C11 and C22?
            # Let's check the formula.
            # C11 = P1 - P4 + P6 - P7.
            # This means the 2x2 matrix P1 is added to the 2x2 block C11.
            # And P1 is also added to C22?
            # C22 = P1 - P3 + P5 + P7.
            # Yes, P1 is added to C22.
            
            # So for each P_k, we have a mapping to the 4 C blocks.
            # P1: C11 (+1), C22 (+1)
            # P2: C21 (+1)
            # P3: C12 (+1), C22 (-1)
            # P4: C11 (-1), C21 (+1)
            # P5: C12 (+1), C22 (+1)
            # P6: C11 (+1)
            # P7: C11 (-1), C22 (+1)
            
            c_block_map = {
                0: [1, 0, 0, 1], # P1 -> C11, C22
                1: [0, 0, 1, 0], # P2 -> C21
                2: [0, 1, 0, -1], # P3 -> C12, C22
                3: [-1, 0, 1, 0], # P4 -> C11, C21
                4: [0, 1, 0, 1], # P5 -> C12, C22
                5: [1, 0, 0, 0], # P6 -> C11
                6: [-1, 0, 0, 1] # P7 -> C11, C22
            }
            
            # We are inside the loop for P_k (index k)
            k = block_products.index((x_blocks, y_blocks))
            c_coeffs = c_block_map[k]
            
            # Now expand w_t (coeffs for 2x2 P_k) to w_full (coeffs for 4x4 C)
            w_full = expand_w(w_t, c_coeffs)
            
            add_term(u_full, v_full, w_full)

    # Now handle B2 (4x4) and B3 (4x1)
    # We need to offset the indices for B and C.
    # B1 is cols 0-3. B2 is cols 4-7. B3 is col 8.
    # C1 is cols 0-3. C2 is cols 4-7. C3 is col 8.
    
    # We already added terms for B1 (cols 0-3).
    # For B2, we shift B indices by 4 and C indices by 4.
    # U is the same.
    
    # Let's refactor to allow offsets.
    
    # Re-implementing with offsets for clarity and correctness.
    
    U = []
    V = []
    W = []
    
    def add_term(u, v, w):
        U.append(u)
        V.append(v)
        W.append(w)
        
    # Strassen 2x2 terms
    strassen_terms = [
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
        ([0,0,1,1], [1,0,0,0], [0,0,1,0]),
        ([1,0,0,0], [0,1,0,-1], [0,1,0,-1]),
        ([0,0,0,1], [0,0,1,-1], [-1,0,1,0]),
        ([1,1,0,0], [0,0,0,1], [0,1,0,1]),
        ([0,0,1,-1], [1,1,0,0], [1,0,0,0]),
        ([0,1,0,-1], [0,0,1,1], [0,0,0,1])
    ]
    
    block_products = [
        ([1,0,0,1], [1,0,0,1]),
        ([0,0,1,1], [1,0,0,0]),
        ([1,0,0,0], [0,1,0,-1]),
        ([0,0,0,1], [0,-1,1,0]),
        ([1,1,0,0], [0,0,0,1]),
        ([-1,0,1,0], [1,1,0,0]),
        ([0,1,0,-1], [0,0,1,1])
    ]
    
    c_block_map = [
        [1, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 0, 0],
        [-1, 0, 0, 1]
    ]
    
    def get_block_indices(block_id):
        if block_id == 0: return [0, 1, 4, 5]
        elif block_id == 1: return [2, 3, 6, 7]
        elif block_id == 2: return [8, 9, 12, 13]
        elif block_id == 3: return [10, 11, 14, 15]
        
    def expand_u(coeff_2x2, block_coeffs):
        u_full = [0.0] * 16
        for i in range(2):
            for j in range(2):
                c = coeff_2x2[i*2+j]
                if c == 0: continue
                for b in range(4):
                    bc = block_coeffs[b]
                    if bc == 0: continue
                    idxs = get_block_indices(b)
                    idx = idxs[i*2+j]
                    u_full[idx] += c * bc
        return u_full

    def expand_v(coeff_2x2, block_coeffs, offset):
        v_full = [0.0] * 36
        for i in range(2):
            for j in range(2):
                c = coeff_2x2[i*2+j]
                if c == 0: continue
                for b in range(4):
                    bc = block_coeffs[b]
                    if bc == 0: continue
                    idxs = get_block_indices(b)
                    idx = idxs[i*2+j] + offset
                    v_full[idx] += c * bc
        return v_full

    def expand_w(coeff_2x2, block_coeffs, offset):
        w_full = [0.0] * 36
        for i in range(2):
            for j in range(2):
                c = coeff_2x2[i*2+j]
                if c == 0: continue
                for b in range(4):
                    bc = block_coeffs[b]
                    if bc == 0: continue
                    idxs = get_block_indices(b)
                    idx = idxs[i*2+j] + offset
                    w_full[idx] += c * bc
        return w_full

    # Add terms for B1 (offset 0)
    for k in range(7):
        x_blocks, y_blocks = block_products[k]
        c_coeffs = c_block_map[k]
        for u_t, v_t, w_t in strassen_terms:
            u_full = expand_u(u_t, x_blocks)
            v_full = expand_v(v_t, y_blocks, 0)
            w_full = expand_w(w_t, c_coeffs, 0)
            add_term(u_full, v_full, w_full)
            
    # Add terms for B2 (offset 4)
    for k in range(7):
        x_blocks, y_blocks = block_products[k]
        c_coeffs = c_block_map[k]
        for u_t, v_t, w_t in strassen_terms:
            u_full = expand_u(u_t, x_blocks)
            v_full = expand_v(v_t, y_blocks, 4)
            w_full = expand_w(w_t, c_coeffs, 4)
            add_term(u_full, v_full, w_full)
            
    # Add terms for B3 (4x1)
    # Rank 4 decomposition.
    # C_j = sum_k A_jk B_k
    # Term j: u_j has 1s at A_j1..A_j4, v_j has 1s at B_1..B_4, w_j has 1 at C_j
    for j in range(4):
        u_full = [0.0] * 16
        v_full = [0.0] * 36
        w_full = [0.0] * 36
        
        # A_jk is at index j*4 + k
        for k in range(4):
            u_full[j*4 + k] = 1.0
            v_full[k] = 1.0 # B is 4x1, indices 0..3. Offset for B3 is 8?
            # B3 is the 9th column.
            # B indices: 0..35.
            # B1: 0..15 (4x4)
            # B2: 16..31 (4x4)
            # B3: 32..35 (4x1)
            # So offset for B3 is 32.
            v_full[k + 32] = 1.0
            
        # C_j is at index j (since C is 4x1? No, C is 4x9).
        # C3 is the 9th column.
        # C indices: 0..35.
        # C1: 0..15
        # C2: 16..31
        # C3: 32..35
        # So C_j is at index j + 32.
        w_full[j + 32] = 1.0
        
        add_term(u_full, v_full, w_full)
        
    return {"U": U, "V": V, "W": W}
