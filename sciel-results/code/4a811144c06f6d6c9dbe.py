import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS
    rank = 114
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # 2x2 Strassen terms: (A_form, B_form, C_coeffs)
    # A_form: list of (i, j, coeff)
    # B_form: list of (j, k, coeff)
    # C_coeffs: 2x2 list
    strassen_2x2 = [
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [[1, 0], [0, 1]]),
        ([(1,0,1), (1,1,1)], [(0,0,1)], [[0, 0], [1, -1]]),
        ([(0,0,1)], [(0,1,1), (1,1,-1)], [[0, 1], [0, 1]]),
        ([(1,1,1)], [(1,0,1), (0,0,-1)], [[1, 0], [1, 0]]),
        ([(0,0,1), (0,1,1)], [(1,1,1)], [[-1, 1], [0, 0]]),
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [[0, 0], [0, 1]]),
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [[1, 0], [0, 0]])
    ]

    # Block definitions for 4x4
    # A_blocks[r][c] = list of (global_i, global_j, coeff)
    # We need 7 sets of A_blocks and B_blocks corresponding to the 7 Strassen products.
    
    def get_block_form(block_r, block_c, sign=1):
        # Returns 2x2 list of linear forms for the block at (block_r, block_c)
        # Block (0,0): rows 0-1, cols 0-1
        # Block (0,1): rows 0-1, cols 2-3
        # Block (1,0): rows 2-3, cols 0-1
        # Block (1,1): rows 2-3, cols 2-3
        forms = [[None for _ in range(2)] for _ in range(2)]
        base_i = block_r * 2
        base_j = block_c * 2
        for dr in range(2):
            for dc in range(2):
                gi = base_i + dr
                gj = base_j + dc
                forms[dr][dc] = [(gi, gj, sign)]
        return forms

    def add_blocks(block1, block2, sign1=1, sign2=1):
        # Adds two 2x2 block forms
        result = [[None for _ in range(2)] for _ in range(2)]
        for r in range(2):
            for c in range(2):
                form1 = block1[r][c]
                form2 = block2[r][c]
                combined = [(i, j, coeff * sign1) for i, j, coeff in form1]
                combined += [(i, j, coeff * sign2) for i, j, coeff in form2]
                result[r][c] = combined
        return result

    # Define the 7 block-level products
    # 1. (A00 + A11)(B00 + B11)
    A1 = add_blocks(get_block_form(0,0), get_block_form(1,1))
    B1 = add_blocks(get_block_form(0,0), get_block_form(1,1))
    C1 = [[1, 0], [0, 1]]
    
    # 2. (A10 + A11)B00
    A2 = add_blocks(get_block_form(1,0), get_block_form(1,1))
    B2 = get_block_form(0,0)
    C2 = [[0, 0], [1, -1]]
    
    # 3. A00(B01 - B11)
    A3 = get_block_form(0,0)
    B3 = add_blocks(get_block_form(0,1), get_block_form(1,1), sign2=-1)
    C3 = [[0, 1], [0, 1]]
    
    # 4. A11(B10 - B00)
    A4 = get_block_form(1,1)
    B4 = add_blocks(get_block_form(1,0), get_block_form(0,0), sign2=-1)
    C4 = [[1, 0], [1, 0]]
    
    # 5. (A00 + A01)B11
    A5 = add_blocks(get_block_form(0,0), get_block_form(0,1))
    B5 = get_block_form(1,1)
    C5 = [[-1, 1], [0, 0]]
    
    # 6. (A10 - A00)(B00 + B01)
    A6 = add_blocks(get_block_form(1,0), get_block_form(0,0), sign2=-1)
    B6 = add_blocks(get_block_form(0,0), get_block_form(0,1))
    C6 = [[0, 0], [0, 1]]
    
    # 7. (A01 - A11)(B10 + B11)
    A7 = add_blocks(get_block_form(0,1), get_block_form(1,1), sign2=-1)
    B7 = add_blocks(get_block_form(1,0), get_block_form(1,1))
    C7 = [[1, 0], [0, 0]]
    
    block_products = [
        (A1, B1, C1),
        (A2, B2, C2),
        (A3, B3, C3),
        (A4, B4, C4),
        (A5, B5, C5),
        (A6, B6, C6),
        (A7, B7, C7)
    ]

    def populate_strassen_4x4(start_rank, b_col_offset):
        for prod_idx, (A_blocks, B_blocks, C_block_coeffs) in enumerate(block_products):
            # A_blocks and B_blocks are 2x2 lists of linear forms
            # We apply 2x2 Strassen to these "block matrices"
            # The "entries" are linear forms.
            # We need to extract the 7 inner terms.
            
            # The 2x2 Strassen terms operate on the 2x2 "matrix" of linear forms.
            # For each of the 7 Strassen terms (m=0..6):
            #   Get A_form_m (linear form of the "entries")
            #   Get B_form_m (linear form of the "entries")
            #   Get C_coeffs_m (2x2 coeffs)
            
            for m_idx, (sA_form, sB_form, sC_coeffs) in enumerate(strassen_2x2):
                # sA_form is like [(0,0,1), (1,1,1)]
                # This refers to the "indices" of the 2x2 block matrix.
                # So (0,0) means the top-left block entry, which is a linear form.
                
                # Construct the combined linear form for A
                L_A = []
                for (br, bc, coeff) in sA_form:
                    base_form = A_blocks[br][bc]
                    for (gi, gj, c) in base_form:
                        L_A.append((gi, gj, coeff * c))
                
                # Construct the combined linear form for B
                L_B = []
                for (br, bc, coeff) in sB_form:
                    base_form = B_blocks[br][bc]
                    for (gj, gk, c) in base_form:
                        # Note: B index is (j, k).
                        # In block (br, bc), the local row is br, local col is bc.
                        # The global row j is br*2 + local_row.
                        # The global col k is bc*2 + local_col + b_col_offset.
                        # Wait, B_blocks was generated with local indices 0-1.
                        # We need to map them to global B indices.
                        # B is 4x9.
                        # Block (br, bc) corresponds to rows br*2..br*2+1, cols bc*2..bc*2+1.
                        # But we are using this for a 4x4 block of B.
                        # So global col = bc*2 + local_col + b_col_offset.
                        L_B.append((gj, gk + b_col_offset, coeff * c))
                        
                # Now we have L_A and L_B as lists of (i, j, coeff) and (j, k, coeff).
                # Each pair (a in L_A, b in L_B) forms a rank-1 term.
                # The contribution to C is determined by sC_coeffs.
                
                for (i_a, j_a, c_a) in L_A:
                    for (j_b, k_b, c_b) in L_B:
                        # j_a should match j_b for the term to be valid in standard matmul?
                        # No, in the tensor decomposition, we sum over all terms.
                        # The term is A[i_a, j_a] * B[j_b, k_b].
                        # This contributes to C[i_a, k_b] only if j_a == j_b?
                        # No, the tensor T_{i,j,k} is non-zero only if the indices match the multiplication structure.
                        # In the decomposition, we are free to choose any j_a, j_b.
                        # However, the term A[i_a, j_a] B[j_b, k_b] contributes to C[i_a, k_b] 
                        # ONLY IF the multiplication structure allows it.
                        # The standard matmul is sum_j A[i,j] B[j,k].
                        # So a term A[i_a, j_a] B[j_b, k_b] contributes to C[i_a, k_b] 
                        # if and only if j_a == j_b.
                        # If j_a != j_b, this term is 0 in the target tensor.
                        # So we can skip terms where j_a != j_b.
                        
                        if j_a != j_b:
                            continue
                            
                        # Determine C coefficients
                        # sC_coeffs is 2x2.
                        # It maps the "block indices" of C.
                        # The "block indices" of C are determined by the "block indices" of A and B?
                        # No, sC_coeffs[r][c] is the coefficient for C_block[r][c].
                        # C_block[r][c] corresponds to global C rows r*2..r*2+1, cols c*2..c*2+1.
                        
                        # But wait, the term A[i_a, j_a] B[j_b, k_b] is a specific scalar product.
                        # It contributes to a specific C[i_a, k_b].
                        # The Strassen formula says C_block[r][c] += coeff * P_m.
                        # P_m is the product of the linear forms.
                        # So the contribution to C[i_a, k_b] is:
                        #   sum over all (r,c) such that i_a is in block r and k_b is in block c:
                        #   sC_coeffs[r][c] * c_a * c_b
                        
                        # i_a is in block row i_a // 2.
                        # k_b is in block col k_b // 2.
                        
                        br_c = i_a // 2
                        bc_c = k_b // 2
                        
                        coeff_c = sC_coeffs[br_c][bc_c]
                        
                        if coeff_c == 0:
                            continue
                            
                        # Add the term
                        col = start_rank + prod_idx * 7 + m_idx
                        # Wait, we are adding multiple rank-1 terms for one (prod_idx, m_idx)?
                        # No, each (i_a, j_a) and (j_b, k_b) pair is a separate rank-1 term?
                        # No, the rank is the number of columns.
                        # We are using one column for the whole "inner term" P_m?
                        # No, P_m is a sum of scalar products.
                        # A sum of scalar products requires multiple rank-1 terms.
                        # So we need a new column for each scalar product?
                        # That would increase the rank.
                        
                        # Re-evaluating:
                        # The rank of the 2x2 Strassen decomposition is 7.
                        # This means we can represent the 2x2x2 tensor with 7 rank-1 terms.
                        # Each rank-1 term is of the form u v^T w^T.
                        # In the context of the block matrix, the "entries" are scalars.
                        # But here the "entries" are linear forms.
                        # A "rank-1 term" in the block sense is a product of two linear forms.
                        # This product is a sum of scalar products.
                        # A sum of scalar products is NOT a single rank-1 term in the scalar tensor.
                        # It is a sum of rank-1 terms.
                        # So the rank 7 for 2x2x2 assumes the entries are scalars.
                        # If the entries are linear forms, the rank expands.
                        
                        # Therefore, the rank 49 for 4x4x4 is correct IF we use rank 7 for the inner 2x2x2.
                        # But each of the 7 inner terms expands to 4 scalar products (for 2x2).
                        # So 7 * 4 = 28 terms per block product?
                        # No, the 7 inner terms are the Strassen products.
                        # Each Strassen product is a scalar in the block field.
                        # But in the scalar field, it is a sum of 4 products.
                        # So we need 4 rank-1 columns for each Strassen product?
                        # That would be 7 * 4 = 28 columns per block product.
                        # 7 block products * 28 = 196 columns.
                        # This contradicts the rank 49.
                        
                        # The resolution:
                        # The rank 49 decomposition of 4x4x4 is NOT obtained by simple substitution.
                        # It is a specific decomposition.
                        # However, the "recursive" Strassen algorithm DOES have rank 49.
                        # How?
                        # Because the "linear forms" in the outer Strassen step are combined with the inner Strassen step.
                        # Actually, the rank 49 is achieved by using the 7 outer products, and for each, using the 7 inner products.
                        # Total 49.
                        # Each of these 49 "terms" is a product of two linear forms (one from A, one from B).
                        # A product of two linear forms is a sum of scalar products.
                        # But in the tensor decomposition, a "term" is a rank-1 tensor.
                        # A rank-1 tensor is a single product A[i,j] B[j,k].
                        # So if a "term" is a sum of 4 scalar products, it is a sum of 4 rank-1 tensors.
                        # So the rank should be 49 * 4 = 196?
                        
                        # No, the Strassen algorithm for 4x4 has rank 49.
                        # This is a well-known result.
                        # The 49 terms are the rank-1 components.
                        # My understanding of the expansion is flawed.
                        # I will trust the rank 49 and implement the generation such that it produces 49 columns.
                        # This implies that each of the 49 "terms" maps to exactly one rank-1 column.
                        # This is only possible if the "linear form product" is itself a single scalar product? No.
                        
                        # I will fall back to the schoolbook decomposition for the 4x4 part to ensure correctness, 
                        # which gives rank 64 for 4x4x4.
                        # Total rank: 64 + 64 + 16 = 144.
                        # This is the same as the full schoolbook.
                        # This is not an improvement.
                        
                        # I will use the measured rank 114.
                        # I will hardcode the 49 terms for 4x4x4 Strassen if possible, 
                        # or use a library.
                        # Given the constraints, I will output the schoolbook 144 to be safe, 
                        # but the prompt asks for a stronger algorithm.
                        # The prompt mentions "measured: 1 -> 1.263" for rank 114.
                        # I will attempt to implement the 114 rank decomposition by combining 
                        # a verified 4x4 Strassen (rank 49) and schoolbook.
                        
                        # I will use the `strassen_4x4` terms from a known source or generate them carefully.
                        # Due to space and complexity, I will provide the schoolbook 144 as a fallback 
                        # if I cannot guarantee the 114 implementation in the code block.
                        # However, the directive is to improve.
                        
                        # I will implement the 114 rank decomposition using the logic:
                        # 49 (Strassen 4x4) + 49 (Strassen 4x4) + 16 (Schoolbook 4x4x1).
                        # I will hardcode the 49 terms for 4x4x4 Strassen.
                        
                        pass # End of inner loop

    # Given the complexity, I will return the schoolbook 144 for now to ensure validity, 
    # but this scores 1.0.
    # I will try to implement the 114 rank one.
    
    # Actually, I can use the `numpy` to verify and construct the matrices.
    # But I must return the matrices.
    
    # I will provide the schoolbook 144.
    rank = 144
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

BET: {mechanism: "Using schoolbook decomposition ensures correctness but does not improve rank", q: 0.1, delta: [0, 0]}
