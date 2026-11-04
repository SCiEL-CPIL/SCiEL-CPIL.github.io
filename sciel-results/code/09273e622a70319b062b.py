import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = 4, 4, 5
    t_start = time.time()
    DEADLINE = 120  # seconds
    
    # Strassen decomposition for 2x2 matrices
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 - M5 + M7 + M4
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6

    # For 4x4x5, we can decompose as:
    # 4x4x4 (using Strassen twice: rank 7*7=49) + 4x4x1 (rank 16) = 65
    # But we can do better by directly applying Strassen-like ideas to 4x4x5.
    
    # Actually, let's just do 4x4x4 with Strassen (49) and 4x4x1 with schoolbook (16).
    # Total rank = 65.
    
    # But wait - can we do 4x4x5 more cleverly?
    # The 4x4x4 part uses Strassen: rank 49.
    # The 4x4x1 part: multiplying 4x4 by 4x1. This is 16 scalar multiplications.
    # Can we reduce this? The tensor for 4x4x1 has rank 16 (it's just the 16 basis tensors).
    # So 49 + 16 = 65 is the best I can easily construct.
    
    # Actually, let me reconsider. For the 4x4x4 part, can I do better than 49?
    # Lander-Bailey gives 19 for 4x4x4. If I can implement it, I'd get 19+16=35.
    # But implementing LB correctly is risky. Let me try a different approach.
    
    # Alternative: Use the fact that 4x4x5 can be viewed as two 4x4x2 problems?
    # No, that doesn't help directly.
    
    # Let me just implement the 65-rank solution (Strassen 49 + schoolbook 16).
    # This is an improvement over 80.
    
    rank = 65
    U = np.zeros((n*m, rank), dtype=np.int8)
    V = np.zeros((m*p, rank), dtype=np.int8)
    W = np.zeros((p*n, rank), dtype=np.int8)
    
    col = 0
    
    # Part 1: 4x4x4 using Strassen (rank 49)
    # We'll implement this by applying Strassen at two levels.
    # Level 1: 4x4 = 2x2 blocks of 2x2 matrices. 7 Strassen products of 2x2 matrices.
    # Level 2: Each 2x2 product uses Strassen: 7 scalar products.
    # Total: 49 scalar products.
    
    # Define the Strassen coefficients for 2x2x2.
    # For a 2x2 matrix, indices are 0,1,2,3 for positions (0,0),(0,1),(1,0),(1,1).
    
    # Strassen terms for 2x2x2:
    # term, u_indices, v_indices, w_indices, signs
    strassen_2x2 = [
        # M1: (A00+A11)(B00+B11) -> contributes to C00 and C11
        ([0, 3], [0, 3], [0, 3], (1, 1, 1, 1)),  # C00 += M1, C11 += M1
        # Actually, let me be more careful.
        # M1 = (A00+A11)(B00+B11)
        # C00 += M1, C11 += M1
        # But M1 is a scalar product of sums, so it's one term in the tensor decomposition.
    ]
    
    # Let me implement this differently. I'll directly construct the 49 terms.
    
    # For the 4x4x4 Strassen decomposition:
    # We have 7 "big" Strassen terms, each of which is a 2x2 matrix product.
    # Each 2x2 product is decomposed into 7 scalar products.
    
    # Big Strassen terms for 4x4:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Where A11, A12, A21, A22 are 2x2 blocks of A (4x4).
    # Similarly for B.
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For each Mi, which is a 2x2 matrix product, we use Strassen to decompose it.
    
    # Let's define a helper to add a Strassen 2x2x2 decomposition.
    
    def add_strassen_2x2(U, V, W, u_idx, v_idx, w_idx, col_start, a_signs, b_signs):
        """Add 7 Strassen terms for a 2x2x2 multiplication.
        u_idx: list of 4 U indices for the 2x2 A block
        v_idx: list of 4 V indices for the 2x2 B block
        w_idx: list of 4 W indices for the 2x2 C block
        a_signs: signs for A entries in the big Strassen term
        b_signs: signs for B entries in the big Strassen term
        """
        c = col_start
        
        # M1: (A00+A11)(B00+B11)
        for ui, si in zip([u_idx[0], u_idx[3]], [a_signs[0], a_signs[3]]):
            for vi, sb in zip([v_idx[0], v_idx[3]], [b_signs[0], b_signs[3]]):
                if si != 0 and sb != 0:
                    # This is a cross term, not a single scalar product
                    pass
        # This approach is getting complicated. Let me use a different method.
        
        return c
    
    # Let me just hardcode the 49 terms directly.
    # This is tedious but reliable.
    
    # Actually, let me try a simpler approach: just use schoolbook for 4x4x4 (64 terms)
    # and schoolbook for 4x4x1 (16 terms). Total 80. No improvement.
    
    # Or: Use Strassen for 4x4x4. Let me carefully implement it.
    
    # I'll create the 49 terms by iterating over the 7 big Strassen terms
    # and for each, the 7 small Strassen terms.
    
    # U indices for 4x4x4 (first 4 columns of B):
    # A[i,j] -> U[4*i + j] for i,j in 0..3
    # Blocks:
    # A11: U[0], U[1], U[4], U[5]  (i=0,1; j=0,1)
    # A12: U[2], U[3], U[6], U[7]  (i=0,1; j=2,3)
    # A21: U[8], U[9], U[12], U[13] (i=2,3; j=0,1)
    # A22: U[10], U[11], U[14], U[15] (i=2,3; j=2,3)
    
    # V indices for 4x4x4 (first 4 columns of B):
    # B[j,k] -> V[4*j + k] for j,k in 0..3
    # Blocks:
    # B11: V[0], V[1], V[4], V[5]  (j=0,1; k=0,1)
    # B12: V[2], V[3], V[6], V[7]  (j=0,1; k=2,3)
    # B21: V[8], V[9], V[12], V[13] (j=2,3; k=0,1)
    # B22: V[10], V[11], V[14], V[15] (j=2,3; k=2,3)
    
    # W indices for 4x4x4 (first 4 rows of C, but C is 4x5):
    # C[i,k] -> W[4*k + i] for i,k in 0..3
    # Blocks:
    # C11: W[0], W[1], W[4], W[5]  (i=0,1; k=0,1)
    # C12: W[8], W[9], W[12], W[13] (i=0,1; k=2,3)
    # C21: W[2], W[3], W[6], W[7]  (i=2,3; k=0,1)
    # C22: W[10], W[11], W[14], W[15] (i=2,3; k=2,3)
    
    # For the 4x4x1 part (5th column of B):
    # B[j,4] -> V[4*j + 4] for j in 0..3
    # C[i,4] -> W[4*4 + i] = W[16+i] for i in 0..3
    
    # Now, the 7 big Strassen terms and their 2x2 block contributions:
    # Each big term Mi = Ai * Bi, where Ai and Bi are 2x2 matrices.
    # Ai and Bi are linear combinations of the 2x2 blocks.
    
    # Let me define the 7 big terms with their A and B block coefficients.
    
    # M1: A = A11 + A22, B = B11 + B22
    # M2: A = A21 + A22, B = B11
    # M3: A = A11, B = B12 - B22
    # M4: A = A22, B = B21 - B11
    # M5: A = A11 + A12, B = B22
    # M6: A = A21 - A11, B = B11 + B12
    # M7: A = A12 - A22, B = B21 + B22
    
    # For each Mi, I need to decompose the 2x2 product Ai*Bi into 7 scalar products
    # using Strassen.
    
    # But Ai and Bi are themselves linear combinations of 2x2 blocks.
    # So I need to expand Ai and Bi into their individual entries.
    
    # This is getting complex. Let me just use the schoolbook for the 4x4x4 part.
    # Rank 64 for 4x4x4 + 16 for 4x4x1 = 80. No improvement.
    
    # Alternatively, I can use a known rank-32 decomposition for 4x4x4?
    # I'm not sure one exists.
    
    # Let me try yet another approach: use the fact that 4x4x5 = 4x4x4 + 4x4x1.
    # For 4x4x4, use Strassen (rank 49). For 4x4x1, use schoolbook (rank 16).
    # Total 65.
    
    # To implement Strassen for 4x4x4, I'll directly enumerate the 49 terms.
    
    # Let me define the Strassen coefficients for 2x2x2.
    # For 2x2 matrices with indices 0,1,2,3:
    # S1 = (A0+A3)(B0+B3)
    # S2 = (A2+A3)B0
    # S3 = A0(B1-B3)
    # S4 = A3(B2-B0)
    # S5 = (A0+A1)B3
    # S6 = (A2-A0)(B0+B1)
    # S7 = (A1-A3)(B2+B3)
    # C0 = S1 + S4 - S5 + S7
    # C1 = S3 + S5
    # C2 = S2 + S4
    # C3 = S1 - S2 + S3 + S6
    
    # Now for each big Strassen term, I'll apply this to the 2x2 blocks.
    
    # Let me create a list of the 49 (u, v, w) index triples with signs.
    
    terms = []
    
    # Helper to get U indices for a 2x2 block of A
    def A_block(i1, j1):
        # i1, j1 in {0,1} for block position
        # Returns [U[4*(2*i1)+2*j1], U[4*(2*i1)+2*j1+1], U[4*(2*i1+1)+2*j1], U[4*(2*i1+1)+2*j1+1]]
        base_i = 2 * i1
        base_j = 2 * j1
        return [4*base_i + base_j, 4*base_i + base_j + 1, 4*(base_i+1) + base_j, 4*(base_i+1) + base_j + 1]
    
    def B_block(j1, k1):
        base_j = 2 * j1
        base_k = 2 * k1
        return [4*base_j + base_k, 4*base_j + base_k + 1, 4*(base_j+1) + base_k, 4*(base_j+1) + base_k + 1]
    
    def C_block(i1, k1):
        base_i = 2 * i1
        base_k = 2 * k1
        return [4*base_k + base_i, 4*base_k + base_i + 1, 4*(base_k+1) + base_i, 4*(base_k+1) + base_i + 1]
    
    # Now, for each of the 7 big Strassen terms, I'll decompose the 2x2 product.
    
    # Big term 1: M1 = (A11+A22)(B11+B22)
    # A part: A11 + A22
    # B part: B11 + B22
    
    # I need to compute (A11+A22) * (B11+B22) using Strassen.
    # Let A' = A11 + A22, B' = B11 + B22.
    # A' entries: A'[0] = A11[0]+A22[0], A'[1] = A11[1]+A22[1], A'[2] = A11[2]+A22[2], A'[3] = A11[3]+A22[3]
    # Similarly for B'.
    
    # Then apply Strassen to A'*B'.
    
    # This is equivalent to expanding (A11+A22)*(B11+B22) = A11*B11 + A11*B22 + A22*B11 + A22*B22.
    # But Strassen doesn't work that way. Strassen works on the combined matrix.
    
    # Let me just directly compute the 7 Strassen terms for A'*B'.
    
    # A' = A11 + A22
    # A'_indices = [(A11[i] + A22[i]) for i in 0..3]
    # In terms of U indices:
    # A11 indices: [0,1,4,5]
    # A22 indices: [10,11,14,15]
    # So A' has entries at U indices [0+10, 1+11, 4+14, 5+15] with signs +1,+1.
    
    # Similarly for B' = B11 + B22:
    # B11 indices: [0,1,4,5]
    # B22 indices: [10,11,14,15]
    # B' has entries at V indices [0+10, 1+11, 4+14, 5+15] with signs +1,+1.
    
    # Now apply Strassen to A'*B'.
    # S1 = (A'[0]+A'[3])(B'[0]+B'[3])
    # This expands to (A11[0]+A22[0]+A11[3]+A22[3])(B11[0]+B22[0]+B11[3]+B22[3])
    # = sum of all products.
    
    # In the tensor decomposition, each scalar product is a separate term.
    # So S1 contributes 4*4 = 16 terms? No, that's not right.
    # S1 is a single scalar product of two sums. In the tensor decomposition,
    # it's represented as a single rank-1 term with U and V being the sums.
    # So U has +1 at the 4 A' indices, V has +1 at the 4 B' indices.
    
    # Wait, that's not how tensor decomposition works. Each term in the decomposition
    # is a rank-1 tensor: U_i * V_j * W_k.
    # For S1 = (A'[0]+A'[3])(B'[0]+B'[3]), this is NOT a single rank-1 term.
    # It's 4 rank-1 terms: A'[0]*B'[0], A'[0]*B'[3], A'[3]*B'[0], A'[3]*B'[3].
    
    # So I need to expand all of these.
    
    # This means the 4x4x4 Strassen decomposition has 7*4 = 28 terms at the first level,
    # but each term is a product of two 2x2 matrices, which expands to 4 scalar products.
    # So 7*4 = 28? No, Strassen has 7 terms, each of which is a 2x2 product.
    # Each 2x2 product is 4 scalar multiplications (schoolbook) or 7 (Strassen).
    
    # I'm confusing myself. Let me just use the schoolbook for 4x4x4.
    # Rank 64. Plus 16 for 4x4x1. Total 80.
    
    # To get an improvement, I need to use a non-schoolbook method.
    # The simplest is Strassen for 4x4x4, which gives rank 49.
    
    # Let me implement Strassen for 4x4x4 by directly enumerating the 49 scalar products.
    
    # I'll use the following approach:
    # 1. Define the 7 big Strassen terms with their A and B block linear combinations.
    # 2. For each big term, expand the 2x2 product into 4 scalar products (schoolbook for the 2x2 level).
    #    Wait, if I use schoolbook for the 2x2 level, I get 7*4 = 28 terms, not 49.
    #    To get 49, I need to use Strassen for the 2x2 level too.
    
    # Let me use Strassen for both levels: 7*7 = 49 terms.
    
    # This is getting too complex to implement correctly without testing.
    # Let me just return the schoolbook (80) to be safe.
    
    # Actually, let me try a different approach. I'll use numpy to verify a candidate
    # decomposition before returning it.
    
    # For now, let me return the schoolbook.
    rank = n * m * p  # 80
    U = np.zeros((n*m, rank), dtype=np.int8)
    V = np.zeros((m*p, rank), dtype=np.int8)
    W = np.zeros((p*n, rank), dtype=np.int8)
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
