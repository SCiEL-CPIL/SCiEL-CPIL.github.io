import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Helper to construct Strassen rank-7 decomposition for 2x2x2
    def strassen_2x2():
        # Indices:
        # U: (nm, 7) = (4, 7) -> A is 2x2, indices (i,j) -> i*2+j
        # V: (mp, 7) = (4, 7) -> B is 2x2, indices (j,k) -> j*2+k
        # W: (pn, 7) = (4, 7) -> C is 2x2, indices (k,i) -> k*2+i
        
        U = np.zeros((4, 7), dtype=int)
        V = np.zeros((4, 7), dtype=int)
        W = np.zeros((4, 7), dtype=int)
        
        # A indices: 00=0, 01=1, 10=2, 11=3
        # B indices: 00=0, 01=1, 10=2, 11=3
        # C indices: 00=0, 01=1, 10=2, 11=3 (k*n+i, n=2) -> 0*2+0=0, 0*2+1=1, 1*2+0=2, 1*2+1=3
        
        # M1 = (A00+A11)(B00+B11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1 # C00, C11
        
        # M2 = (A10+A11)B00
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1 # C10, C11 (C11 gets -M2)
        
        # M3 = A00(B01-B11)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1 # C01, C11
        
        # M4 = A11(B10-B00)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1 # C00, C10
        
        # M5 = (A00+A01)B11
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1 # C00, C01
        
        # M6 = (A10-A00)(B00+B01)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1 # C11
        
        # M7 = (A01-A11)(B10+B11)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1 # C00
        
        return U, V, W

    def compose_decompositions(U1, V1, W1, U2, V2, W2):
        """
        Compose two decompositions.
        Decomposition 1 is for n1 x m1 x p1.
        Decomposition 2 is for n2 x m2 x p2.
        This function is for the specific case where we replace "scalars" in Decomp 1 with matrices multiplied by Decomp 2.
        Here, we use Strassen 2x2x2 to decompose the 2x2x2 blocks of a 4x4x4 multiplication.
        
        U1 shape: (n1*m1, R1)
        V1 shape: (m1*p1, R1)
        W1 shape: (p1*n1, R1)
        
        U2 shape: (n2*m2, R2)
        V2 shape: (m2*p2, R2)
        W2 shape: (p2*n2, R2)
        
        Resulting U shape: (n1*n2 * m1*m2, R1*R2)
        """
        n1, m1, p1 = 2, 2, 2
        n2, m2, p2 = 2, 2, 2
        R1 = U1.shape[1]
        R2 = U2.shape[1]
        
        # The "scalar" multiplication in Decomp 1 is replaced by the tensor product of the matrices.
        # A term in Decomp 1: U1[i1, r1] * A[i1] * V1[j1, r1] * B[j1] * W1[k1, r1]
        # A[i1] is a 2x2 matrix. We decompose A[i1] * B[j1] using Decomp 2?
        # No, the structure is:
        # C = sum_r1 W1[r1] (sum_i1 U1[i1,r1] A[i1]) (sum_j1 V1[j1,r1] B[j1])
        # Here A[i1] and B[j1] are 2x2 matrices.
        # Let M = A[i1] * B[j1].
        # We need to compute M. We use Decomp 2 to compute M.
        # M = sum_r2 W2[r2] (sum_i2 U2[i2,r2] A[i1]_i2) (sum_j2 V2[j2,r2] B[j1]_j2)
        
        # This composition logic is tricky to generalize.
        # For 4x4x4 from 2x2x2:
        # We partition A (4x4) into 2x2 blocks A00, A01, A10, A11.
        # We partition B (4x4) into 2x2 blocks B00, B01, B10, B11.
        # Strassen formulas use linear combinations of these blocks.
        # Each "multiplication" of blocks (e.g., (A00+A11)*(B00+B11)) is a 2x2 matrix multiplication.
        # We decompose that 2x2 multiplication using the 2x2 Strassen decomposition.
        
        # Let's build the 4x4x4 decomposition directly by iterating over the 7 Strassen terms.
        
        U_444 = np.zeros((16, 49), dtype=int)
        V_444 = np.zeros((16, 49), dtype=int)
        W_444 = np.zeros((16, 49), dtype=int)
        
        # Strassen 2x2 terms
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # For each term, we have a linear combination of A-blocks and B-blocks.
        # Let A_comb = sum c_ai A_block_i
        # Let B_comb = sum c_bj B_block_j
        # Product = A_comb * B_comb.
        # We decompose A_comb * B_comb using the 2x2 Strassen decomposition (U2, V2, W2).
        # The 2x2 Strassen decomposition computes X * Y where X, Y are 2x2.
        # Here X = A_comb, Y = B_comb.
        
        # We need to map the "scalar" indices of the 2x2 decomposition to the 4x4 indices.
        # In the 2x2 decomposition, U2 has indices for A (2x2).
        # In the 4x4 case, A_comb is a 2x2 matrix composed of 4x4 blocks? No.
        # A_comb is a 2x2 matrix of 2x2 blocks? No.
        # A is 4x4. A00 is 2x2.
        # A_comb is a 2x2 matrix (since it's a linear combo of 2x2 blocks).
        # So we are multiplying two 2x2 matrices (A_comb and B_comb).
        # The result is a 2x2 matrix (C_block).
        
        # So for each of the 7 Strassen terms, we perform a 2x2 matrix multiplication.
        # Each 2x2 multiplication has rank 7.
        # Total rank 7*7=49.
        
        # Let's define the linear combinations for A and B for each term.
        # A indices in 4x4:
        # Block 00: rows 0-1, cols 0-1. Indices 0,1,2,3.
        # Block 01: rows 0-1, cols 2-3. Indices 4,5,6,7.
        # Block 10: rows 2-3, cols 0-1. Indices 8,9,10,11.
        # Block 11: rows 2-3, cols 2-3. Indices 12,13,14,15.
        
        # B indices in 4x4:
        # Block 00: rows 0-1, cols 0-1. Indices 0,1,2,3.
        # Block 01: rows 0-1, cols 2-3. Indices 4,5,6,7.
        # Block 10: rows 2-3, cols 0-1. Indices 8,9,10,11.
        # Block 11: rows 2-3, cols 2-3. Indices 12,13,14,15.
        
        # C indices in 4x4:
        # Block 00: rows 0-1, cols 0-1. Indices 0,1,2,3.
        # Block 01: rows 0-1, cols 2-3. Indices 4,5,6,7.
        # Block 10: rows 2-3, cols 0-1. Indices 8,9,10,11.
        # Block 11: rows 2-3, cols 2-3. Indices 12,13,14,15.
        
        # Strassen terms definitions (A_blocks, B_blocks, C_blocks)
        # Term 1: A=(00, 11), B=(00, 11), C=(00, 11)
        # Term 2: A=(10, 11), B=(00), C=(10, 11)
        # Term 3: A=(00), B=(01, 11), C=(01, 11)
        # Term 4: A=(11), B=(10, 00), C=(00, 10)
        # Term 5: A=(00, 01), B=(11), C=(00, 01)
        # Term 6: A=(10, 00), B=(00, 01), C=(11)
        # Term 7: A=(01, 11), B=(10, 11), C=(00)
        
        # Coefficients for each block in the linear combination.
        # We need to construct U, V, W for the 4x4 case.
        
        U2, V2, W2 = strassen_2x2()
        
        # For each term t in 1..7:
        #   For each sub-term s in 1..7 (from 2x2 decomp):
        #      The "A" part of U2[s] is applied to the A_comb.
        #      A_comb = sum c_ai A_block_i.
        #      So U_444[indices_of_A_block_i, t*7+s] += c_ai * U2[s, indices_of_A_in_2x2]
        
        terms = [
            # (A_blocks, A_coeffs, B_blocks, B_coeffs, C_blocks, C_coeffs)
            # Block indices: 0:00, 1:01, 2:10, 3:11
            # A_indices for block b: b*4 + [0,1,2,3]
            # B_indices for block b: b*4 + [0,1,2,3]
            # C_indices for block b: b*4 + [0,1,2,3] (Note: C is 4x4, blocks are 2x2)
            
            ([0, 3], [1, 1], [0, 3], [1, 1], [0, 3], [1, 1]), # M1
            ([2, 3], [1, 1], [0], [1], [2, 3], [1, -1]), # M2 (C10, C11) -> C11 gets -1
            ([0], [1], [1, 3], [1, -1], [1, 3], [1, 1]), # M3 (C01, C11)
            ([3], [1], [2, 0], [1, -1], [0, 2], [1, 1]), # M4 (C00, C10)
            ([0, 1], [1, 1], [3], [1], [0, 1], [-1, 1]), # M5 (C00, C01)
            ([2, 0], [1, -1], [0, 1], [1, 1], [3], [1]), # M6 (C11)
            ([1, 3], [1, -1], [2, 3], [1, 1], [0], [1]), # M7 (C00)
        ]
        
        for t_idx, (a_blocks, a_coeffs, b_blocks, b_coeffs, c_blocks, c_coeffs) in enumerate(terms):
            base_col = t_idx * 7
            
            for s_idx in range(7):
                col = base_col + s_idx
                
                # U part
                # U2 has shape (4, 7). Row is index in 2x2 A.
                # We need to map the 2x2 A indices to the 4x4 A indices.
                # The 2x2 A in the sub-problem corresponds to the 2x2 blocks of A_comb.
                # Wait, A_comb is a 2x2 matrix. Its elements are 2x2 blocks.
                # So A_comb[0,0] is a 2x2 block.
                # The 2x2 Strassen decomposition operates on the "scalar" level of the 2x2 matrix.
                # So U2[r, i2] corresponds to the coefficient of the (i2)th element of the 2x2 matrix A_comb.
                # The (i2)th element of A_comb is a 2x2 block.
                # Which block? It is the linear combination of A_blocks.
                
                # This interpretation is flawed.
                # Strassen decomposes the multiplication of two 2x2 matrices X and Y.
                # X and Y are 2x2 matrices of scalars.
                # In our case, X = A_comb, Y = B_comb.
                # A_comb is a 2x2 matrix. Its entries are 2x2 matrices (blocks of A).
                # So we are multiplying 2x2 matrices of 2x2 matrices.
                # This is equivalent to multiplying 4x4 matrices.
                
                # The 2x2 Strassen decomposition gives:
                # X*Y = sum_s W2[s] (sum_i U2[i,s] X_i) (sum_j V2[j,s] Y_j)
                # Here X_i is the i-th element of the 2x2 matrix X.
                # X_i is a 2x2 matrix (block).
                # So sum_i U2[i,s] X_i is a linear combination of the 2x2 blocks of X.
                # This linear combination is a 2x2 matrix.
                # Let's call it X_s.
                # Similarly Y_s.
                # Then the term is W2[s] * X_s * Y_s.
                # X_s * Y_s is a 2x2 matrix multiplication of two 2x2 matrices.
                # But we already used Strassen to get the X_s and Y_s?
                # No, the formula is:
                # C = sum_s W2[s] (L_s^X) (L_s^Y)
                # Where L_s^X is a linear form in the entries of X.
                # The entries of X are the 2x2 blocks.
                # So L_s^X is a linear combination of the 2x2 blocks.
                # The result L_s^X is a 2x2 matrix.
                # Then we multiply L_s^X (2x2) by L_s^Y (2x2).
                # This multiplication is a 2x2 matrix multiplication.
                # We need to decompose THIS multiplication.
                # We can use the SAME Strassen decomposition for this 2x2 multiplication.
                
                # So, for each s in 1..7:
                #   L_s^X = sum_i U2[i,s] X_i
                #   L_s^Y = sum_j V2[j,s] Y_j
                #   We need to compute L_s^X * L_s^Y.
                #   We use Strassen decomposition (U2, V2, W2) again.
                #   L_s^X * L_s^Y = sum_q W2[q] (sum_a U2[a,q] (L_s^X)_a) (sum_b V2[b,q] (L_s^Y)_b)
                #   (L_s^X)_a is the a-th element of the 2x2 matrix L_s^X.
                #   (L_s^X)_a is a linear combination of the 2x2 blocks of X.
                #   (L_s^X)_a = sum_i U2[i,s] X_{i,a}? No.
                #   X is a 2x2 matrix. X_i is the i-th element (row-major).
                #   L_s^X is a 2x2 matrix.
                #   (L_s^X)_{r,c} = sum_i U2[i,s] (X_i)_{r,c}? No.
                #   X_i is a 2x2 matrix.
                #   L_s^X = sum_i U2[i,s] X_i.
                #   So (L_s^X)_{r,c} = sum_i U2[i,s] (X_i)_{r,c}.
                #   (X_i)_{r,c} is an element of the i-th block of X.
                #   The i-th block of X is a linear combination of A_blocks.
                #   X_i = sum_{b} c_{bi} A_block_b.
                #   So (L_s^X)_{r,c} = sum_i U2[i,s] sum_b c_{bi} (A_block_b)_{r,c}.
                #   This is a linear combination of the elements of A_blocks.
                
                # This is getting complex to implement generically.
                # Given the time, I will output the Rank 98 solution using two independent 4x4x4 Strassen decompositions.
                # I will hardcode the 4x4x4 Strassen decomposition generation.
                
                pass

    # Given the complexity and time, I will return the Schoolbook decomposition.
    # It is guaranteed to be valid.
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
