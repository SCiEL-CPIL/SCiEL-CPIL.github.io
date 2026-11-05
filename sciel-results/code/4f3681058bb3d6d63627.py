"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 30: 2 * 7 (Strassen for 4x4x4) + 16 (Schoolbook for 4x4x1)
    rank = 30
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a rank-1 term
    def add_term(i, j, k, col, u_val, v_val, w_val):
        U[i * m + j][col] = u_val
        V[j * p + k][col] = v_val
        W[k * n + i][col] = w_val

    # Strassen for 4x4x4 on B[:, 0:4]
    def strassen_4x4x4(offset_k, col_offset):
        # M1 = (A11 + A22)(B11 + B22)
        # A11 is indices i in 0,1, j in 0,1. A22 is i in 2,3, j in 2,3.
        # B11 is j in 0,1, k in 0,1 (relative). B22 is j in 2,3, k in 2,3 (relative).
        
        # M1
        for i in [0, 1, 2, 3]:
            for j in [0, 1, 2, 3]:
                if (i < 2 and j < 2) or (i >= 2 and j >= 2):
                    # A part is 1
                    # B part: k in 0,1,2,3 relative
                    for k_rel in [0, 1, 2, 3]:
                        if (k_rel < 2 and (j < 2)) or (k_rel >= 2 and (j >= 2)):
                             # Wait, B11 is j in 0,1 and k in 0,1. B22 is j in 2,3 and k in 2,3.
                             # The term is (A11+A22) * (B11+B22).
                             # This is a product of two 2x2 matrices.
                             # It expands to 4 terms.
                             pass
        # This manual expansion is error-prone. Let's use the known Strassen factors for 2x2x2 and compose.
        # Actually, it's easier to just write out the 7 products for 4x4x4 directly.
        
        # The 7 products for 4x4x4 using 2x2 blocks:
        # Let A = [[A11, A12], [A21, A22]] and B = [[B11, B12], [B21, B22]]
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # We need to map these to U, V, W.
        # U corresponds to A, V to B, W to C.
        # Each M is a 4x4 matrix product of sums/diffs of 2x2 blocks.
        # This means each M contributes to multiple entries in U, V, W.
        
        # Let's define the 2x2 blocks of A (indices i,j) and B (indices j,k)
        # A11: i in {0,1}, j in {0,1}
        # A12: i in {0,1}, j in {2,3}
        # A21: i in {2,3}, j in {0,1}
        # A22: i in {2,3}, j in {2,3}
        
        # B11: j in {0,1}, k in {0,1}
        # B12: j in {0,1}, k in {2,3}
        # B21: j in {2,3}, k in {0,1}
        # B22: j in {2,3}, k in {2,3}
        
        # M1 = (A11+A22)(B11+B22)
        # This is a rank-1 term in the tensor? No, it's a product of two sums.
        # (A11+A22) is a 4x4 matrix with 1s in A11 and A22 blocks.
        # (B11+B22) is a 4x4 matrix with 1s in B11 and B22 blocks.
        # The product contributes to C11, C12, C21, C22.
        # Specifically, C11 gets M1, C22 gets M1.
        # C12 gets nothing from M1? No, M1 is the full product.
        # Wait, Strassen computes 7 products M1..M7 which are 2x2 matrices.
        # Then C blocks are linear combos of M's.
        # So M1 is a 4x4 matrix? No, M1 is a 2x2 matrix product of 2x2 blocks.
        # So M1 is a 2x2 matrix.
        # The tensor decomposition rank is 7 for the 2x2x2 case.
        # For 4x4x4, we are doing 7 products of 2x2 matrices.
        # Each 2x2 matrix product has rank 4 (schoolbook) or can be decomposed further?
        # No, the standard Strassen for 4x4x4 uses rank 7 * 4 = 28?
        # No, the rank of the 4x4x4 tensor is 7 if we consider the 2x2 blocks as scalars?
        # No, the entries are scalars.
        # The rank of the 4x4x4 tensor is 7 * 4 = 28?
        # Let's check. The 2x2x2 tensor has rank 7.
        # The 4x4x4 tensor is a block matrix of 2x2x2 tensors.
        # If we use Strassen, we replace 8 scalar multiplications with 7.
        # But each "scalar" is a 2x2 matrix.
        # So we do 7 matrix multiplications of 2x2 matrices.
        # Each 2x2 matrix multiplication has rank 4.
        # So total rank is 7 * 4 = 28.
        # This is better than 64 (schoolbook 4x4x4).
        # So for 4x4x9:
        # Two 4x4x4 parts: 2 * 28 = 56.
        # One 4x4x1 part: 16.
        # Total rank: 56 + 16 = 72.
        # This is better than 144.
        
        # Let's implement Strassen for 4x4x4 with rank 28.
        pass

    # Re-evaluating rank:
    # 2x2x2 Strassen rank is 7.
    # 4x4x4 using Strassen on 2x2 blocks:
    # The 4x4x4 tensor T satisfies T[i,j,k] = sum_c U[i,c] V[j,c] W[k,c].
    # If we use Strassen, we form 7 intermediate 2x2 matrices M1..M7.
    # Each Mq is a product of two 2x2 matrices (sums of blocks of A and B).
    # So Mq = Pq * Qq where Pq, Qq are 2x2 matrices.
    # The product Pq * Qq can be decomposed into 4 rank-1 terms (schoolbook for 2x2x2).
    # So each Mq contributes 4 to the rank.
    # Total rank for 4x4x4 is 7 * 4 = 28.
    
    # For 4x4x9:
    # Split B into B1 (4x4), B2 (4x4), B3 (4x1).
    # C1 = A B1 (4x4), C2 = A B2 (4x4), C3 = A B3 (4x1).
    # Rank(C1) = 28.
    # Rank(C2) = 28.
    # Rank(C3) = 16 (schoolbook 4x4x1).
    # Total rank = 28 + 28 + 16 = 72.
    
    # This is a valid improvement over 144.
    
    rank = 72
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_rank1(i, j, k, col, u_val, v_val, w_val):
        U[i * m + j][col] = u_val
        V[j * p + k][col] = v_val
        W[k * n + i][col] = w_val

    # Strassen 4x4x4 implementation
    def strassen_4x4x4(k_offset, col_offset):
        # k_offset: 0 for B1, 4 for B2
        # col_offset: 0 for C1, 28 for C2
        
        # The 7 Strassen products for 2x2 blocks:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # We need to expand each Mq into 4 rank-1 terms for the 2x2 matrix multiplication.
        # Let P be the A-side matrix (2x2) and Q be the B-side matrix (2x2).
        # P = [[p00, p01], [p10, p11]], Q = [[q00, q01], [q10, q11]].
        # Product R = P Q.
        # R00 = p00*q00 + p01*q10
        # R01 = p00*q01 + p01*q11
        # R10 = p10*q00 + p11*q10
        # R11 = p10*q01 + p11*q11
        
        # Each term p_a * q_b is a rank-1 contribution to the 2x2x2 tensor.
        # We need to map the 2x2 block indices to the 4x4 indices.
        
        # A blocks:
        # A11: i in {0,1}, j in {0,1}
        # A12: i in {0,1}, j in {2,3}
        # A21: i in {2,3}, j in {0,1}
        # A22: i in {2,3}, j in {2,3}
        
        # B blocks (relative to k_offset):
        # B11: j in {0,1}, k in {0,1}
        # B12: j in {0,1}, k in {2,3}
        # B21: j in {2,3}, k in {0,1}
        # B22: j in {2,3}, k in {2,3}
        
        # Define a helper to add a 2x2 matrix product P*Q to the tensor.
        # P is defined by its 4 values (coefficients for A blocks)
        # Q is defined by its 4 values (coefficients for B blocks)
        # Actually, P and Q are linear combinations of A and B blocks.
        # So P[i,j] is a coefficient for A[i,j].
        # Q[j,k] is a coefficient for B[j,k].
        
        def add_2x2_product(P, Q, col_base):
            # P is a 2x2 matrix of coefficients for the A blocks.
            # P[0,0] corresponds to A11, P[0,1] to A12, etc.
            # Q is a 2x2 matrix of coefficients for the B blocks.
            # Q[0,0] corresponds to B11, Q[0,1] to B12, etc.
            
            # The product is sum_{a,b} P[a,b] * Q[c,d] * A_block[a] * B_block[c]
            # This is not quite right.
            # P is a 4x4 matrix derived from A. Q is a 4x4 matrix derived from B.
            # P[i,j] = sum_{block} coeff_P[block, i_block, j_block] * A[i,j]
            # This is getting complicated.
            
            # Simpler: Just iterate over all i, j, k in 0..3.
            # For a given term in the Strassen formula, we have specific coefficients for each A[i,j] and B[j,k].
            
            # Let's define the coefficients for A and B for each of the 7 Strassen terms.
            # Term 1: (A11+A22)(B11+B22)
            # A coeff: 1 for A11, A22; 0 for A12, A21.
            # B coeff: 1 for B11, B22; 0 for B12, B21.
            
            # We will define 7 functions that return the A-coeff matrix (4x4) and B-coeff matrix (4x4).
            
            def get_coeffs(term_id):
                A_coeff = [[0]*4 for _ in range(4)]
                B_coeff = [[0]*4 for _ in range(4)]
                
                if term_id == 1:
                    # A11 + A22
                    for i in [0,1]:
                        for j in [0,1]: A_coeff[i][j] = 1
                    for i in [2,3]:
                        for j in [2,3]: A_coeff[i][j] = 1
                    # B11 + B22
                    for j in [0,1]:
                        for k in [0,1]: B_coeff[j][k] = 1
                    for j in [2,3]:
                        for k in [2,3]: B_coeff[j][k] = 1
                elif term_id == 2:
                    # (A21+A22) B11
                    for i in [2,3]:
                        for j in [0,1]: A_coeff[i][j] = 1
                        for j in [2,3]: A_coeff[i][j] = 1
                    for j in [0,1]:
                        for k in [0,1]: B_coeff[j][k] = 1
                elif term_id == 3:
                    # A11 (B12 - B22)
                    for i in [0,1]:
                        for j in [0,1]: A_coeff[i][j] = 1
                    for j in [0,1]:
                        for k in [2,3]: B_coeff[j][k] = 1
                    for j in [2,3]:
                        for k in [2,3]: B_coeff[j][k] = -1
                elif term_id == 4:
                    # A22 (B21 - B11)
                    for i in [2,3]:
                        for j in [2,3]: A_coeff[i][j] = 1
                    for j in [2,3]:
                        for k in [0,1]: B_coeff[j][k] = 1
                    for j in [0,1]:
                        for k in [0,1]: B_coeff[j][k] = -1
                elif term_id == 5:
                    # (A11+A12) B22
                    for i in [0,1]:
                        for j in [0,1]: A_coeff[i][j] = 1
                        for j in [2,3]: A_coeff[i][j] = 1
                    for j in [2,3]:
                        for k in [2,3]: B_coeff[j][k] = 1
                elif term_id == 6:
                    # (A21-A11) (B11+B12)
                    for i in [2,3]:
                        for j in [0,1]: A_coeff[i][j] = 1
                    for i in [0,1]:
                        for j in [0,1]: A_coeff[i][j] = -1
                    for j in [0,1]:
                        for k in [0,1]: B_coeff[j][k] = 1
                        for k in [2,3]: B_coeff[j][k] = 1
                elif term_id == 7:
                    # (A12-A22) (B21+B22)
                    for i in [0,1]:
                        for j in [2,3]: A_coeff[i][j] = 1
                    for i in [2,3]:
                        for j in [2,3]: A_coeff[i][j] = -1
                    for j in [2,3]:
                        for k in [0,1]: B_coeff[j][k] = 1
                        for k in [2,3]: B_coeff[j][k] = 1
                        
                return A_coeff, B_coeff

            # For each term, we add 4 rank-1 contributions corresponding to the 2x2 matrix multiplication.
            # Wait, the product of two 4x4 matrices (which are sparse) is not rank 4.
            # The product of two sparse matrices can have higher rank.
            # However, the Strassen decomposition works on the *blocks*.
            # The rank of the 4x4x4 tensor using Strassen is 28.
            # This means we can decompose it into 28 rank-1 tensors.
            # The 7 Strassen terms are 7 "block" multiplications.
            # Each block multiplication is a 2x2x2 tensor with rank 4 (schoolbook).
            # So 7 * 4 = 28.
            
            # So for each term_id (1..7), we get A_coeff (4x4) and B_coeff (4x4).
            # These define the factors for the 7th "block".
            # We then need to decompose the product of these two 4x4 matrices into 4 rank-1 terms?
            # No, the 4x4 matrices are *not* arbitrary. They are block-diagonal or block-structured.
            # Actually, A_coeff is a 4x4 matrix where each 2x2 block is a scalar multiple of the identity?
            # No, A_coeff[i][j] is 1 or -1 or 0.
            # The product C = A_coeff * B_coeff is a 4x4 matrix.
            # We want to express the tensor T[i,j,k] = A[i,j] B[j,k] as sum_c U[i,c] V[j,c] W[k,c].
            # For a fixed term_id, we have a contribution to T.
            # The contribution is (A_coeff * A) * (B_coeff * B)? No.
            # The Strassen formula says:
            # C_block = sum_q coeff_q * M_q
            # M_q = P_q A Q_q B (where P_q, Q_q are block matrices)
            # This is not a rank-1 decomposition of the original tensor.
            # It is a decomposition into 7 products of linear combinations.
            # Each product (P_q A) (Q_q B) is a matrix product.
            # To decompose (P_q A) (Q_q B) into rank-1 terms of the original tensor,
            # we need to expand (P_q A) and (Q_q B).
            # (P_q A) is a 4x4 matrix. (Q_q B) is a 4x4 matrix.
            # Their product is a 4x4 matrix.
            # The rank of the product of two rank-4 matrices is at most 4.
            # So each M_q contributes at most 4 to the rank.
            # Total rank 7 * 4 = 28.
            
            # So we can iterate over the 4 rank-1 terms of the 4x4x4 tensor for each M_q.
            # How to find the 4 rank-1 terms?
            # We can use the schoolbook decomposition for the 4x4x4 tensor, but restricted to the support of A_coeff and B_coeff.
            # Actually, it's easier to just use the 28 rank-1 terms directly.
            
            # Let's just use the 28 rank-1 terms from the standard Strassen 4x4x4 decomposition.
            # I will hardcode the 28 terms.
            
            pass

    # This is getting too complex to generate dynamically without errors.
    # I will return the schoolbook decomposition for now, as it is guaranteed to be correct.
    # Rank 144.
    
    # Wait, I can try a simpler block decomposition.
    # Split 4x4x9 into 2x2x9 blocks?
    # Rank of 2x2x9 is 2*2*9 = 36? No, 2x2x2 is rank 7.
    # 2x2x9 = 2x2x8 + 2x2x1.
    # 2x2x8 = 4 * 2x2x2 = 4 * 7 = 28.
    # 2x2x1 = 4.
    # Total for 2x2x9 is 32.
    # 4x4x9 = 4 blocks of 2x2x9?
    # A is 2x2 blocks of 2x2. B is 2x4 blocks of 2x2?
    # This is getting complicated.
    
    # Let's stick to the schoolbook for correctness.
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

# EVOLVE-BLOCK-END
