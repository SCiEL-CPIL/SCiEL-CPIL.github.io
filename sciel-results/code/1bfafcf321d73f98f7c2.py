"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=6
    
    # We will build the decomposition for 4x4x4 (rank 49) and 4x4x2 (rank 28)
    
    U_4x4x4 = []
    V_4x4x4 = []
    W_4x4x4 = []
    
    U_4x4x2 = []
    V_4x4x2 = []
    W_4x4x2 = []
    
    # --- Helper: Base case 2x2x2 Strassen ---
    # Returns U, V, W as lists of lists (4x7)
    def strassen_2x2x2():
        # U, V, W are 4x7
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # Indices in 2x2 matrix:
        # 0: (0,0)  1: (0,1)
        # 2: (1,0)  3: (1,1)
        
        # P1 = (A11 + A22)(B11 + B22)
        # A: 0, 3. B: 0, 3. C: 0 (C11), 3 (C22)
        U[:, 0] = [1, 0, 0, 1]
        V[:, 0] = [1, 0, 0, 1]
        W[:, 0] = [1, 0, 0, 1] # C11 (+1), C22 (+1)
        
        # P2 = (A21 + A22)B11
        # A: 2, 3. B: 0. C: 1 (C21), 3 (C22)
        U[:, 1] = [0, 0, 1, 1]
        V[:, 1] = [1, 0, 0, 0]
        W[:, 1] = [0, 1, 0, -1] # C21 (+1), C22 (-1)
        
        # P3 = A11(B12 - B22)
        # A: 0. B: 1, 3. C: 1 (C12), 3 (C22)
        U[:, 2] = [1, 0, 0, 0]
        V[:, 2] = [0, 1, 0, -1]
        W[:, 2] = [0, 1, 0, 1] # C12 (+1), C22 (+1)
        
        # P4 = A22(B21 - B11)
        # A: 3. B: 2, 0. C: 1 (C21), 0 (C11)
        U[:, 3] = [0, 0, 0, 1]
        V[:, 3] = [0, 0, 1, -1]
        W[:, 3] = [1, 1, 0, 0] # C11 (+1), C21 (+1)
        
        # P5 = (A11 + A12)B22
        # A: 0, 1. B: 3. C: 0 (C11), 1 (C12)
        U[:, 4] = [1, 1, 0, 0]
        V[:, 4] = [0, 0, 0, 1]
        W[:, 4] = [-1, 1, 0, 0] # C11 (-1), C12 (+1)
        
        # P6 = (A21 - A11)(B11 + B12)
        # A: 2, 0. B: 0, 1. C: 3 (C22)
        U[:, 5] = [-1, 0, 1, 0]
        V[:, 5] = [1, 1, 0, 0]
        W[:, 5] = [0, 0, 0, 1] # C22 (+1)
        
        # P7 = (A12 - A22)(B21 + B22)
        # A: 1, 3. B: 2, 3. C: 0 (C11), 3 (C22)
        U[:, 6] = [0, 1, 0, -1]
        V[:, 6] = [0, 0, 1, 1]
        W[:, 6] = [1, 0, 0, 1] # C11 (+1), C22 (+1)
        
        return U, V, W

    # --- Helper: Embed 2x2x2 into 4x4x4 ---
    # offset_i, offset_j: block position in A (0 or 1)
    # offset_k: block position in B (0 or 1) for cols, and C (0 or 1) for rows?
    # In 4x4x4, A is 4x4, B is 4x4, C is 4x4.
    # Blocks are 2x2.
    # A block (bi, bj) corresponds to rows 2*bi..2*bi+1, cols 2*bj..2*bj+1.
    # B block (bj, bk) corresponds to rows 2*bj..2*bj+1, cols 2*bk..2*bk+1.
    # C block (bk, bi) corresponds to rows 2*bk..2*bk+1, cols 2*bi..2*bi+1.
    
    def embed_2x2_in_4x4(U_loc, V_loc, W_loc, bi, bj, bk):
        # U: 4x7 -> 16x7
        # V: 4x7 -> 16x7
        # W: 4x7 -> 16x7
        
        U_global = np.zeros((16, 7))
        V_global = np.zeros((16, 7))
        W_global = np.zeros((16, 7))
        
        for i_loc in range(2):
            for j_loc in range(2):
                i_g = 2 * bi + i_loc
                j_g = 2 * bj + j_loc
                idx_u = i_g * 4 + j_g
                U_global[idx_u, :] = U_loc[i_loc * 2 + j_loc, :]
                
        for j_loc in range(2):
            for k_loc in range(2):
                j_g = 2 * bj + j_loc
                k_g = 2 * bk + k_loc
                idx_v = j_g * 4 + k_g
                V_global[idx_v, :] = V_loc[j_loc * 2 + k_loc, :]
                
        for k_loc in range(2):
            for i_loc in range(2):
                k_g = 2 * bk + k_loc
                i_g = 2 * bi + i_loc
                idx_w = k_g * 4 + i_g
                W_global[idx_w, :] = W_loc[k_loc * 2 + i_loc, :]
                
        return U_global, V_global, W_global

    # --- Construct 4x4x4 (Rank 49) ---
    U_444_list = []
    V_444_list = []
    W_444_list = []
    
    U_base, V_base, W_base = strassen_2x2x2()
    
    # Strassen 4x4x4 uses 7 products of 2x2x2
    # P1 = (A11+A22)(B11+B22) -> A blocks (0,0),(1,1); B blocks (0,0),(1,1)
    # This requires summing decompositions.
    # It's easier to just iterate through the 7 Strassen steps for the 4x4 case.
    
    # Step 1: P1
    # A11 (0,0) + A22 (1,1)
    # B11 (0,0) + B22 (1,1)
    # This is not a single 2x2 multiplication.
    # We must decompose the bilinear form.
    
    # Actually, the recursive Strassen works by treating the 2x2 blocks as scalar entries.
    # The rank is 7 * (rank of 2x2x2).
    # So we can just take the 7 terms of the "outer" Strassen, and for each term,
    # if it involves a sum of blocks, we distribute.
    
    # Let's do it explicitly.
    # Outer Strassen for 4x4:
    # M1 = (A11 + A22) (B11 + B22)
    # M2 = (A21 + A22) B11
    # M3 = A11 (B12 - B22)
    # M4 = A22 (B21 - B11)
    # M5 = (A11 + A12) B22
    # M6 = (A21 - A11) (B11 + B12)
    # M7 = (A12 - A22) (B21 + B22)
    
    # Each Mi is a product of two 2x2 matrices.
    # We need to decompose each Mi into rank 7.
    
    def add_decomposition(U_list, V_list, W_list, U_prod, V_prod, W_prod):
        U_list.append(U_prod)
        V_list.append(V_prod)
        W_list.append(W_prod)

    # Helper to get block from 4x4 matrix representation in decomposition
    # We are building the decomposition, not the matrices.
    # So we need to construct U, V, W for the product of specific blocks.
    
    # Product of A_block(bi, bj) and B_block(bj, bk)
    # This is a 2x2x2 multiplication.
    # We use the base case decomposition, embedded.
    
    # M1: (A11 + A22)(B11 + B22)
    # = A11B11 + A11B22 + A22B11 + A22B22
    # We add 4 decompositions.
    
    # A11 (0,0) * B11 (0,0) -> C11 (0,0) part?
    # In 4x4, C11 gets contribution from M1, M4, M5, M7.
    # Wait, the Strassen formulas for C blocks are:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # So for each M_i, we determine which C blocks it contributes to, and with what sign.
    # Then we decompose the product of A_blocks and B_blocks.
    
    def add_block_product(U_list, V_list, W_list, bi, bj, bk, sign_c11, sign_c12, sign_c21, sign_c22):
        # Product A(bi,bj) * B(bj,bk)
        # This is a 2x2x2 multiplication.
        # We get base U, V, W.
        U_b, V_b, W_b = strassen_2x2x2()
        
        # Embed into 4x4 space for A and B
        U_emb, V_emb, W_emb = embed_2x2_in_4x4(U_b, V_b, W_b, bi, bj, bk)
        
        # Now W_emb corresponds to the C block at (bk, bi).
        # But we need to distribute to C11, C12, C21, C22 based on signs.
        # The C block (bk, bi) is a 2x2 matrix.
        # C11 is block (0,0). C12 is (0,1). C21 is (1,0). C22 is (1,1).
        
        # If bk=0, bi=0, this product contributes to C11.
        # If bk=0, bi=1, this product contributes to C12.
        # If bk=1, bi=0, this product contributes to C21.
        # If bk=1, bi=1, this product contributes to C22.
        
        # So we just apply the sign to the W matrix.
        
        W_scaled = W_emb * sign_c11 if (bk==0 and bi==0) else \
                   W_emb * sign_c12 if (bk==0 and bi==1) else \
                   W_emb * sign_c21 if (bk==1 and bi==0) else \
                   W_emb * sign_c22 if (bk==1 and bi==1) else \
                   np.zeros((16, 7))
                   
        # Wait, this logic is flawed.
        # The product A(bi,bj) * B(bj,bk) results in a 2x2 matrix which is exactly the C block (bk, bi).
        # So if we are computing M1 = (A11+A22)(B11+B22), it contributes to C11 and C22?
        # No, M1 is a single 2x2 matrix product? No, M1 is a sum of products.
        # M1 = A11B11 + A11B22 + A22B11 + A22B22.
        # A11B11 is a 2x2 matrix. It is the C11 block of the product A*B?
        # No, in the 4x4 product, C11 = A11B11 + A12B21.
        # The Strassen M1 is (A11+A22)(B11+B22).
        # This M1 is added to C11 and C22.
        # So the 2x2 matrix result of (A11+A22)(B11+B22) is added to C11 and C22.
        
        # So, for the term A11B11 within M1:
        # It contributes to the C11 block and C22 block?
        # No. The matrix M1 is added to C11 and C22.
        # So the 2x2 matrix M1 is added to C11 (top-left) and C22 (bottom-right).
        
        # Therefore, the decomposition of the product A11B11 should have its W part
        # placed in the C11 block AND the C22 block?
        # Yes.
        
        # So, for a specific product A(bi,bj) * B(bj,bk):
        # It generates a 2x2 matrix P.
        # This P is part of M_i.
        # M_i contributes to certain C blocks with certain signs.
        # So P is added to those C blocks.
        
        # So we need to embed the W part into the correct C blocks.
        
        # Let's re-embed W.
        # W_emb currently has the result in the (bk, bi) block of a 4x4 C matrix?
        # No, embed_2x2_in_4x4 put the result in the (bk, bi) block.
        # If the product is A11B11, bi=0, bj=0, bk=0.
        # The result is in C block (0,0).
        # But M1 (which contains A11B11) is added to C11 (block 0,0) AND C22 (block 1,1).
        # So we need to copy the W vector to block (1,1) as well.
        
        # This means we can't just use a single embed. We need to sum contributions.
        
        # Let's refine `add_block_product`.
        # It should take the list of (target_block, sign) pairs.
        
        pass

    # Let's rewrite the 4x4x4 construction more carefully.
    
    U_444 = []
    V_444 = []
    W_444 = []
    
    # Define the 7 Strassen terms for 4x4
    # Each term is a sum of products of 2x2 blocks.
    # Term 1: (A11+A22)(B11+B22) -> A11B11 + A11B22 + A22B11 + A22B22
    # Term 2: (A21+A22)B11 -> A21B11 + A22B11
    # Term 3: A11(B12-B22) -> A11B12 - A11B22
    # Term 4: A22(B21-B11) -> A22B21 - A22B11
    # Term 5: (A11+A12)B22 -> A11B22 + A12B22
    # Term 6: (A21-A11)(B11+B12) -> A21B11 + A21B12 - A11B11 - A11B12
    # Term 7: (A12-A22)(B21+B22) -> A12B21 + A12B22 - A22B21 - A22B22
    
    # Contribution of each term to C blocks:
    # C11: +M1 +M4 -M5 +M7
    # C12: +M3 +M5
    # C21: +M2 +M4
    # C22: +M1 -M2 +M3 +M6
    
    # We iterate through each product A(bi,bj)B(bj,bk) with a coefficient.
    # And add its decomposition to U, V, W.
    
    def add_term_to_444(bi, bj, bk, coeff, target_c_blocks):
        # target_c_blocks: dict { (bk, bi): sign }? No, the product is in block (bk, bi) of the "local" C.
        # But this product is part of M_i.
        # M_i is added to specific C blocks.
        # So the 2x2 matrix result is added to those C blocks.
        
        U_b, V_b, W_b = strassen_2x2x2()
        
        # Embed A and B parts
        U_emb, V_emb, _ = embed_2x2_in_4x4(U_b, V_b, W_b, bi, bj, bk)
        
        # W parts: We need to place the 2x2 result into the target C blocks.
        W_total = np.zeros((16, 7))
        for (c_row, c_col), sign in target_c_blocks.items():
            # c_row, c_col are block indices (0 or 1)
            # The local result is a 2x2 matrix.
            # We embed it into the (c_row, c_col) block of the 4x4 C.
            # Note: embed_2x2_in_4x4 for W uses bk, bi.
            # Here we want to put it in (c_row, c_col).
            # So we call embed with bk=c_row, bi=c_col.
            _, _, W_part = embed_2x2_in_4x4(U_b, V_b, W_b, c_col, bj, c_row) 
            # Wait, embed_2x2_in_4x4 signature: (U, V, W, bi, bj, bk).
            # It uses bi, bj for A, bj, bk for B, bk, bi for C.
            # So for C block (c_row, c_col), we need bk=c_row, bi=c_col.
            # But A and B are fixed.
            # So we can't just reuse the embed function for W if the target block is different.
            
            # We need a helper to embed W into a specific block.
            def embed_W_only(W_loc, c_row, c_col):
                W_global = np.zeros((16, 7))
                for k_loc in range(2):
                    for i_loc in range(2):
                        k_g = 2 * c_row + k_loc
                        i_g = 2 * c_col + i_loc
                        idx_w = k_g * 4 + i_g
                        W_global[idx_w, :] = W_loc[k_loc * 2 + i_loc, :]
                return W_global
            
            W_part = embed_W_only(W_b, c_row, c_col)
            W_total += sign * W_part * coeff
            
        U_444.append(U_emb * coeff)
        V_444.append(V_emb * coeff)
        W_444.append(W_total)

    # Term 1: A11B11 (+1), A11B22 (+1), A22B11 (+1), A22B22 (+1)
    # Targets: C11 (+1), C22 (+1)
    add_term_to_444(0, 0, 0, 1, {(0,0): 1, (1,1): 1})
    add_term_to_444(0, 0, 1, 1, {(0,0): 1, (1,1): 1}) # A11 B22 (bj=0, bk=1)
    add_term_to_444(1, 0, 0, 1, {(0,0): 1, (1,1): 1}) # A22 B11 (bi=1, bj=0)
    add_term_to_444(1, 0, 1, 1, {(0,0): 1, (1,1): 1}) # A22 B22 (bi=1, bj=0, bk=1)
    
    # Term 2: A21B11 (+1), A22B11 (+1)
    # Targets: C21 (+1)
    add_term_to_444(1, 0, 0, 1, {(1,0): 1})
    add_term_to_444(1, 0, 0, 1, {(1,0): 1}) # Wait, A22 is bi=1, bj=0. B11 is bj=0, bk=0.
    # A21 is bi=1, bj=0? No. A21 is block (1,0). A22 is block (1,1).
    # M2 = (A21 + A22) B11.
    # A21: bi=1, bj=0.
    # A22: bi=1, bj=1.
    # B11: bj=0, bk=0.
    # So A21B11: bi=1, bj=0, bk=0.
    # A22B11: bi=1, bj=1, bk=0.
    
    # Let's restart the term definitions carefully.
    # A blocks: A11(0,0), A12(0,1), A21(1,0), A22(1,1).
    # B blocks: B11(0,0), B12(0,1), B21(1,0), B22(1,1).
    
    # M1 = (A11+A22)(B11+B22)
    # A11(0,0) B11(0,0) -> bi=0, bj=0, bk=0
    # A11(0,0) B22(1,1) -> bi=0, bj=0, bk=1 (Wait, B22 is bj=1, bk=1. A11 is bj=0. Mismatch!)
    
    # Ah, Strassen requires compatible blocks.
    # A11 (0,0) multiplies B11 (0,0) and B21 (1,0)? No.
    # In matrix multiplication C = AB:
    # C11 = A11B11 + A12B21
    # C12 = A11B12 + A12B22
    # C21 = A21B11 + A22B21
    # C22 = A21B12 + A22B22
    
    # Strassen M1 = (A11+A22)(B11+B22)
    # This expands to A11B11 + A11B22 + A22B11 + A22B22.
    # A11B22: A11 is (0,0), B22 is (1,1). bj of A is 0, bj of B is 1. Mismatch.
    # So this product is NOT a valid block product in the standard sense?
    # No, Strassen works on the *values*.
    # A11 and A22 are 2x2 matrices. B11 and B22 are 2x2 matrices.
    # The sum A11+A22 is a 2x2 matrix.
    # The product (A11+A22)B11 is a valid 2x2 matrix product.
    # But in the tensor decomposition, we are decomposing the bilinear form.
    # The term A11B22 does not exist in the standard block multiplication.
    # However, Strassen's algorithm *defines* new products.
    # So we *must* decompose the product of the summed matrices.
    # (A11+A22) is a 2x2 matrix. B11 is a 2x2 matrix.
    # Their product is a 2x2 matrix.
    # We can decompose this product using the 2x2x2 Strassen.
    # The "A" part of the decomposition will have 1s in the A11 and A22 blocks.
    # The "B" part will have 1s in the B11 block.
    
    # So, for M1 term A11B11 part:
    # A blocks: 0,0 and 1,1.
    # B blocks: 0,0.
    # This is not a single block product.
    
    # My `add_term_to_444` assumed a single block product.
    # I need to handle sums of blocks in A or B.
    
    # Let's generalize.
    # A_term: list of (bi, bj, coeff)
    # B_term: list of (bj, bk, coeff)
    # Target C blocks: list of (c_row, c_col, sign)
    
    def add_general_term(A_blocks, B_blocks, C_targets):
        # A_blocks: [(bi, bj, coeff), ...]
        # B_blocks: [(bj, bk, coeff), ...]
        # C_targets: [(c_row, c_col, sign), ...]
        
        # The product is sum_{a in A_blocks, b in B_blocks} coeff_a * coeff_b * A(a) B(b)
        # This is a sum of block products.
        # We can just call add_term_to_444 for each pair?
        # Yes, because the decomposition is linear.
        
        for (bi, bj, ca) in A_blocks:
            for (bj_b, bk, cb) in B_blocks:
                if bj != bj_b:
                    continue # Invalid product
                add_term_to_444(bi, bj, bk, ca * cb, {t[0]: t[1] for t in C_targets})

    # M1: A: [(0,0,1), (1,1,1)], B: [(0,0,1), (1,1,1)]
    # Targets: C11(0,0,1), C22(1,1,1)
    add_general_term([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)])
    
    # M2: A: [(1,0,1), (1,1,1)], B: [(0,0,1)]
    # Targets: C21(1,0,1)
    add_general_term([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1)])
    
    # M3: A: [(0,0,1)], B: [(0,1,1), (1,1,-1)]
    # Targets: C12(0,1,1)
    add_general_term([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1)])
    
    # M4: A: [(1,1,1)], B: [(1,0,1), (0,0,-1)]
    # Targets: C11(0,0,1), C21(1,0,1)
    add_general_term([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)])
    
    # M5: A: [(0,0,1), (0,1,1)], B: [(1,1,1)]
    # Targets: C11(0,0,-1), C12(0,1,1)
    add_general_term([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)])
    
    # M6: A: [(1,0,1), (0,0,-1)], B: [(0,0,1), (0,1,1)]
    # Targets: C22(1,1,1)
    add_general_term([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)])
    
    # M7: A: [(0,1,1), (1,1,-1)], B: [(1,0,1), (1,1,1)]
    # Targets: C11(0,0,1), C22(1,1,1)
    add_general_term([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1), (1,1,1)])

    U_4x4x4 = np.hstack(U_444)
    V_4x4x4 = np.hstack(V_444)
    W_4x4x4 = np.hstack(W_444)
    
    # --- Construct 4x4x2 (Rank 28) ---
    # A is 4x4. B_right is 4x2.
    # B_right = [B11', B21'] stacked.
    # B11' is 2x2, B21' is 2x2.
    # C_right = [C11', C21'] stacked.
    
    # C11' = A11 B11' + A12 B21'
    # C21' = A21 B11' + A22 B21'
    
    U_442 = []
    V_442 = []
    W_442 = []
    
    def add_term_to_442(bi, bj, bk, coeff, target_c_blocks):
        # bk is 0 or 1 (row block of B_right)
        # target_c_blocks: {(c_row, c_col): sign}
        # c_row is 0 or 1 (row block of C_right)
        # c_col is 0 (only one col block in 4x2? No, C is 4x2.
        # Wait, C is 4x6. C_right is 4x2.
        # C_right has 2 cols.
        # So C11' is 2x2. C21' is 2x2.
        # So C_right is 4x2.
        # The "blocks" in C_right are 2x2.
        # So c_col is always 0?
        # No, C_right has 1 column of 2x2 blocks.
        # So c_col=0.
        
        U_b, V_b, W_b = strassen_2x2x2()
        
        # Embed A
        U_emb, _, _ = embed_2x2_in_4x4(U_b, V_b, W_b, bi, bj, 0) # bk doesn't matter for A
        # Embed B
        # B is 4x2.
        # B_right block (bk, 0) is at rows 2*bk..2*bk+1, cols 0..1.
        # V global indices: j * 6 + k?
        # No, V is (m*p, rank) = (4*6, rank) = (24, rank).
        # Index j*6 + k.
        # For B_right, k is 4,5.
        # So we need to map local k (0,1) to global k (4,5).
        
        V_emb = np.zeros((24, 7))
        for j_loc in range(2):
            for k_loc in range(2):
                j_g = 2 * bj + j_loc
                k_g = 4 + k_loc # B_right starts at col 4
                idx_v = j_g * 6 + k_g
                V_emb[idx_v, :] = V_b[j_loc * 2 + k_loc, :]
                
        # Embed W
        # W is (p*n, rank) = (6*4, rank) = (24, rank).
        # Index k*n + i = k*4 + i.
        # C_right is cols 4,5.
        # C11' is rows 0,1, cols 4,5.
        # C21' is rows 2,3, cols 4,5.
        
        W_total = np.zeros((24, 7))
        for (c_row, c_col), sign in target_c_blocks.items():
            # c_row is 0 or 1.
            # c_col is 0 (only one col block).
            # The local result is 2x2.
            # We embed it into C block (c_row, 0) of C_right.
            # Which corresponds to global rows 2*c_row..2*c_row+1, cols 4..5.
            
            def embed_W_442(W_loc, c_row):
                W_global = np.zeros((24, 7))
                for k_loc in range(2):
                    for i_loc in range(2):
                        k_g = 4 + k_loc
                        i_g = 2 * c_row + i_loc
                        idx_w = k_g * 4 + i_g
                        W_global[idx_w, :] = W_loc[k_loc * 2 + i_loc, :]
                return W_global
            
            W_part = embed_W_442(W_b, c_row)
            W_total += sign * W_part * coeff
            
        U_442.append(U_emb * coeff)
        V_442.append(V_emb * coeff)
        W_442.append(W_total)

    # C11' = A11 B11' + A12 B21'
    # A11 (0,0) B11' (0,0) -> bi=0, bj=0, bk=0
    add_term_to_442(0, 0, 0, 1, {(0,0): 1})
    # A12 (0,1) B21' (1,0) -> bi=0, bj=1, bk=1
    add_term_to_442(0, 1, 1, 1, {(0,0): 1})
    
    # C21' = A21 B11' + A22 B21'
    # A21 (1,0) B11' (0,0) -> bi=1, bj=0, bk=0
    add_term_to_442(1, 0, 0, 1, {(1,0): 1})
    # A22 (1,1) B21' (1,0) -> bi=1, bj=1, bk=1
    add_term_to_442(1, 1, 1, 1, {(1,0): 1})
    
    U_4x4x2 = np.hstack(U_442)
    V_4x4x2 = np.hstack(V_442)
    W_4x4x2 = np.hstack(W_442)
    
    # Combine
    U = np.hstack([U_4x4x4, U_4x4x2])
    V = np.hstack([V_4x4x4, V_4x4x2])
    W = np.hstack([W_4x4x4, W_4x4x2])
    
    # Convert to lists of lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
