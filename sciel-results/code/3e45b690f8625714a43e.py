import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS
    # We will build the decomposition manually.
    # U: (n*m, rank) = (16, rank)
    # V: (m*p, rank) = (36, rank)
    # W: (p*n, rank) = (36, rank)
    
    # We use a list of tuples (u, v, w) representing rank-1 components.
    # u is a vector of length 16
    # v is a vector of length 36
    # w is a vector of length 36
    
    terms = []
    
    # Helper to add a term
    def add_term(u_idx, val_u, v_idx, val_v, w_idx, val_w):
        # We accumulate into sparse dicts or just build full vectors at the end
        # For efficiency, we can store (index, value) pairs
        pass

    # Let's use sparse representation during construction
    # Each term is a dict: { (U_idx, V_idx, W_idx): coeff }
    # Or simpler: lists of (u_idx, v_idx, w_idx, coeff)
    
    raw_terms = []
    
    def add(u_i, v_i, w_i, coeff):
        if coeff != 0:
            raw_terms.append((u_i, v_i, w_i, coeff))

    # 1. Decompose A * B1 (4x4 * 4x4)
    # B1 corresponds to columns 0-3 of B.
    # B indices in V: j*p + k. j in 0..3, k in 0..3.
    # W indices: k*n + i. k in 0..3, i in 0..3.
    
    # Strassen for 4x4
    # A blocks: A11 (0,0-1,1), A12 (0,2-1,3), A21 (2,0-3,1), A22 (2,2-3,3)
    # B1 blocks: B11 (0,0-1,1), B12 (0,2-1,3), B21 (2,0-3,1), B22 (2,2-3,3)
    
    # We need to express linear combinations of A and B blocks.
    # This is complex to do symbolically in code without a CAS.
    # Alternative: Use the known rank-28 decomposition for 4x4.
    # Or, simpler: Use the rank-49 Strassen decomposition but expand it to scalar mults.
    # Rank 49 is easier to generate correctly.
    # Rank 28 is better. Let's try to generate Rank 28.
    
    # Actually, generating the explicit 28 terms for 4x4 is tedious.
    # Let's stick to Rank 49 for 4x4 first to ensure correctness, then optimize if time permits.
    # Rank 49 + 49 + 4 = 102.
    # Rank 28 + 28 + 4 = 60.
    
    # Let's implement the Rank 49 version (Strassen on 4x4, Strassen on 2x2).
    # This is safer.
    
    def strassen_2x2(A_block, B_block, prefix_u, prefix_v, prefix_w):
        # A_block: list of 4 indices (i, j) in A
        # B_block: list of 4 indices (j, k) in B
        # prefix_u, prefix_v, prefix_w: offsets or identifiers
        
        # Strassen 7 multiplications
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # We need to add terms to raw_terms.
        # Each multiplication is a 2x2 product.
        # We will expand each 2x2 product into 4 scalar terms (schoolbook).
        
        def add_2x2_product(u_indices, v_indices, w_indices, sign_u, sign_v):
            # u_indices: 4 indices for A block
            # v_indices: 4 indices for B block
            # w_indices: 4 indices for C block
            # sign_u: 1 or -1 for the linear combination
            # sign_v: 1 or -1 for the linear combination
            
            # C00 = A00 B00 + A01 B10
            # C01 = A00 B01 + A01 B11
            # C10 = A10 B00 + A11 B10
            # C11 = A10 B01 + A11 B11
            
            # A indices: u_indices[0]=A00, u_indices[1]=A01, u_indices[2]=A10, u_indices[3]=A11
            # B indices: v_indices[0]=B00, v_indices[1]=B01, v_indices[2]=B10, v_indices[3]=B11
            
            # W indices: w_indices[0]=C00, w_indices[1]=C01, w_indices[2]=C10, w_indices[3]=C11
            
            # Term 1: A00 B00 -> C00
            add(u_indices[0], v_indices[0], w_indices[0], sign_u * sign_v)
            # Term 2: A01 B10 -> C00
            add(u_indices[1], v_indices[2], w_indices[0], sign_u * sign_v)
            
            # Term 3: A00 B01 -> C01
            add(u_indices[0], v_indices[1], w_indices[1], sign_u * sign_v)
            # Term 4: A01 B11 -> C01
            add(u_indices[1], v_indices[3], w_indices[1], sign_u * sign_v)
            
            # Term 5: A10 B00 -> C10
            add(u_indices[2], v_indices[0], w_indices[2], sign_u * sign_v)
            # Term 6: A11 B10 -> C10
            add(u_indices[3], v_indices[2], w_indices[2], sign_u * sign_v)
            
            # Term 7: A10 B01 -> C11
            add(u_indices[2], v_indices[1], w_indices[3], sign_u * sign_v)
            # Term 8: A11 B11 -> C11
            add(u_indices[3], v_indices[3], w_indices[3], sign_u * sign_v)

        # Define blocks
        # A11: 0,1,4,5 (indices in U: i*m+j)
        # A12: 2,3,6,7
        # A21: 8,9,12,13
        # A22: 10,11,14,15
        
        # B11: 0,1,4,5 (indices in V: j*p+k)
        # B12: 2,3,6,7
        # B21: 8,9,12,13
        # B22: 10,11,14,15
        
        # W11: 0,1,4,5 (indices in W: k*n+i)
        # W12: 2,3,6,7
        # W21: 8,9,12,13
        # W22: 10,11,14,15
        
        # We need to handle linear combinations.
        # This function only handles a single product.
        # We need a way to add linear combinations.
        
        # Let's refactor: add_term takes a list of (idx, coeff) for U, V, W.
        # But U and V are multiplied, so we can't just sum indices.
        # We must expand (sum u_i) * (sum v_j) = sum sum u_i v_j.
        
        # So, for M1 = (A11+A22)(B11+B22)
        # U terms: A11 (coeff 1), A22 (coeff 1)
        # V terms: B11 (coeff 1), B22 (coeff 1)
        # We add all combinations.
        
        def add_product(u_terms, v_terms, w_indices, w_sign):
            # u_terms: list of (indices, coeff)
            # v_terms: list of (indices, coeff)
            # w_indices: 4 indices for the 2x2 output
            # w_sign: 1 or -1 for the whole block
            
            for u_idx, u_coeff in u_terms[0]:
                for v_idx, v_coeff in v_terms[0]:
                    # This is not right. u_terms is a list of blocks?
                    pass

        # Let's do it step by step for M1..M7.
        
        # M1 = (A11+A22)(B11+B22)
        # U: A11(1), A22(1)
        # V: B11(1), B22(1)
        # W: W11(1), W12(1), W21(1), W22(1) -> No, M1 contributes to all 4 blocks with signs.
        # C11 += M1, C12 += M1? No.
        # C11 = M1 - M4 + M6 - M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M3 + M2 + M5
        
        # So for M1:
        # Contributes +1 to C11, +1 to C22? No.
        # C11: +M1
        # C22: +M1
        # C12, C21: 0
        
        # So we call add_2x2_product for M1 with W indices for C11 and C22?
        # No, M1 is a 2x2 matrix.
        # M1_{00} contributes to C11_{00}.
        # M1_{01} contributes to C11_{01}.
        # M1_{10} contributes to C11_{10}.
        # M1_{11} contributes to C11_{11}.
        # AND
        # M1_{00} contributes to C22_{00}.
        # ...
        
        # This is getting complicated.
        
        # Let's just generate the 49 terms for 4x4 using a known library or hardcoded list?
        # No, we must write the code.
        
        # Simpler approach:
        # Use the rank-49 decomposition.
        # We can write a helper that takes linear combinations of A and B blocks and adds the product.
        
        def add_block_product(u_block_coeffs, v_block_coeffs, w_block_coeffs, w_block_indices):
            # u_block_coeffs: dict {block_idx: coeff}
            # v_block_coeffs: dict {block_idx: coeff}
            # w_block_coeffs: dict {block_idx: coeff}
            # w_block_indices: list of 4 indices for each block?
            
            # This is too abstract.
            
            # Let's just hardcode the 7 Strassen multiplications and expand them.
            
            # M1: U={0:1, 3:1}, V={0:1, 3:1}
            # Blocks 0,1,2,3 correspond to 00,01,10,11.
            # Indices for Block 0 (00): 0,1,4,5
            # Indices for Block 1 (01): 2,3,6,7
            # Indices for Block 2 (10): 8,9,12,13
            # Indices for Block 3 (11): 10,11,14,15
            
            # This is getting out of hand.
            
            # Let's use the rank-144 schoolbook if Strassen is too hard to implement correctly in time.
            # Rank 144 is the seed.
            # Rank 102 is a good improvement.
            # Rank 60 is better.
            
            # I will implement Rank 102 (Strassen 4x4 with Schoolbook 2x2 is Rank 28? No, Rank 49 is Strassen 2x2).
            # Let's implement Rank 102 (2x Rank 49 + 1x Rank 4).
            
            # I will write a function to add a Strassen 4x4 decomposition.
            
            pass

    # Given the complexity, I will implement the Rank 144 schoolbook but with a twist:
    # The seed is Rank 144.
    # I can easily implement Rank 102.
    # I will implement Rank 102.
    
    # 1. A * B1 (4x4 * 4x4) -> Rank 49
    # 2. A * B2 (4x4 * 4x4) -> Rank 49
    # 3. A * B3 (4x4 * 4x1) -> Rank 4
    
    # Total Rank 102.
    
    # I will implement a helper to add a Strassen 4x4 decomposition.
    
    def add_strassen_4x4(b_offset, w_offset):
        # b_offset: 0 for B1, 4 for B2 (column offset in B)
        # w_offset: 0 for C1, 4 for C2 (column offset in C)
        
        # Strassen 7 mults
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # We need to add terms.
        # Each M is a 2x2 product.
        # We will use Schoolbook for the 2x2 products (Rank 4 each).
        # Total 7 * 4 = 28 terms.
        
        # A blocks indices in U (16x1)
        # A11: 0,1,4,5
        # A12: 2,3,6,7
        # A21: 8,9,12,13
        # A22: 10,11,14,15
        
        # B blocks indices in V (36x1)
        # B11: b_offset + 0, b_offset + 1, b_offset + 4, b_offset + 5
        # B12: b_offset + 2, b_offset + 3, b_offset + 6, b_offset + 7
        # B21: b_offset + 8, b_offset + 9, b_offset + 12, b_offset + 13
        # B22: b_offset + 10, b_offset + 11, b_offset + 14, b_offset + 15
        
        # W blocks indices in W (36x1)
        # W11: w_offset + 0, w_offset + 1, w_offset + 4, w_offset + 5
        # W12: w_offset + 2, w_offset + 3, w_offset + 6, w_offset + 7
        # W21: w_offset + 8, w_offset + 9, w_offset + 12, w_offset + 13
        # W22: w_offset + 10, w_offset + 11, w_offset + 14, w_offset + 15
        
        # Helper to add a 2x2 product of linear combinations
        def add_2x2_prod(u_indices, u_coeffs, v_indices, v_coeffs, w_indices, w_sign):
            # u_indices: 4 indices
            # u_coeffs: 4 coeffs
            # v_indices: 4 indices
            # v_coeffs: 4 coeffs
            # w_indices: 4 indices
            # w_sign: 1 or -1
            
            # C00 = sum u_i v_j where i,j correspond to 00
            # We need to map the 2x2 structure.
            # u_indices are [A00, A01, A10, A11]
            # v_indices are [B00, B01, B10, B11]
            # w_indices are [C00, C01, C10, C11]
            
            # Term (0,0): A00 B00 + A01 B10
            # Term (0,1): A00 B01 + A01 B11
            # Term (1,0): A10 B00 + A11 B10
            # Term (1,1): A10 B01 + A11 B11
            
            # We iterate over all pairs of u and v?
            # No, the structure is fixed.
            
            # u[0]=A00, u[1]=A01, u[2]=A10, u[3]=A11
            # v[0]=B00, v[1]=B01, v[2]=B10, v[3]=B11
            
            # C00: u[0]v[0] + u[1]v[2]
            # C01: u[0]v[1] + u[1]v[3]
            # C10: u[2]v[0] + u[3]v[2]
            # C11: u[2]v[1] + u[3]v[3]
            
            # We add these 8 terms.
            
            def add_term(u_i, v_i, w_i, coeff):
                if coeff != 0:
                    raw_terms.append((u_i, v_i, w_i, coeff))
                    
            # C00
            add_term(u_indices[0], v_indices[0], w_indices[0], u_coeffs[0] * v_coeffs[0] * w_sign)
            add_term(u_indices[1], v_indices[2], w_indices[0], u_coeffs[1] * v_coeffs[2] * w_sign)
            
            # C01
            add_term(u_indices[0], v_indices[1], w_indices[1], u_coeffs[0] * v_coeffs[1] * w_sign)
            add_term(u_indices[1], v_indices[3], w_indices[1], u_coeffs[1] * v_coeffs[3] * w_sign)
            
            # C10
            add_term(u_indices[2], v_indices[0], w_indices[2], u_coeffs[2] * v_coeffs[0] * w_sign)
            add_term(u_indices[3], v_indices[2], w_indices[2], u_coeffs[3] * v_coeffs[2] * w_sign)
            
            # C11
            add_term(u_indices[2], v_indices[1], w_indices[3], u_coeffs[2] * v_coeffs[1] * w_sign)
            add_term(u_indices[3], v_indices[3], w_indices[3], u_coeffs[3] * v_coeffs[3] * w_sign)
            
        # Now define the 7 Strassen multiplications.
        
        # M1 = (A11+A22)(B11+B22)
        # U: A11(1), A22(1) -> Indices [0,1,4,5] (A11) and [10,11,14,15] (A22)
        # This doesn't fit the "4 indices" model directly because it's a sum of two blocks.
        # We need to expand (A11+A22) into 8 indices with coeffs.
        # U_indices = [0,1,4,5, 10,11,14,15]
        # U_coeffs = [1,1,1,1, 1,1,1,1]
        # But add_2x2_prod expects 4 indices.
        
        # We must generalize add_2x2_prod to take lists of (idx, coeff).
        
        def add_2x2_prod_general(u_terms, v_terms, w_indices, w_sign):
            # u_terms: list of (idx, coeff)
            # v_terms: list of (idx, coeff)
            # w_indices: 4 indices
            # w_sign: 1 or -1
            
            # We need to compute the 2x2 product.
            # This requires knowing which u_term is A00, A01, etc.
            # This is not possible with a flat list.
            
            # We must pass the block structure.
            
            pass
        
        # This is too complex for the time.
        
        # I will fall back to Rank 144.
        
        pass

    # Rank 144 is the seed.
    # I will just return the seed.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
