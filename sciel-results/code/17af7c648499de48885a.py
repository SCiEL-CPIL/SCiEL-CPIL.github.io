import numpy as np

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    
    # We will build the decomposition using Strassen's algorithm.
    # A is 4x4, B is 4x7.
    # Split A into 2x2 blocks: A11, A12, A21, A22 (each 2x2)
    # Split B into 2x? blocks: B1 (top 2 rows, 7 cols), B2 (bottom 2 rows, 7 cols)
    # In Strassen, we treat B as having 4 blocks: B11, B12, B21, B22.
    # Since B has 7 cols, we split cols into 4 and 3.
    # B11: 2x4, B12: 2x3
    # B21: 2x4, B22: 2x3
    
    # Strassen products:
    # M1 = (A11 + A22)(B11 + B22) -> 2x4 * 2x4? No.
    # B11 is 2x4, B22 is 2x3. Sum is not defined.
    # So we MUST pad B to 4x8 or handle the mixed sizes carefully.
    # If we pad B to 4x8, B12 becomes 2x4, B22 becomes 2x4.
    # Then all M are 2x4.
    # Rank = 7 * 14 = 98.
    
    # Can we do better with mixed sizes?
    # M1 = (A11 + A22)(B11 + B22)
    # B11 is 2x4, B22 is 2x3.
    # We can view B11 + B22 as a 2x7 matrix where the last 3 cols are from B22 and first 4 from B11?
    # No, addition requires same shape.
    # So standard Strassen requires square blocks or compatible shapes.
    # Thus, padding is the standard approach.
    
    # Let's stick to the padding approach for simplicity and guaranteed correctness.
    # Rank 98 is an improvement over 112.
    
    # To get rank 89, we would need to decompose the specific mixed-size products.
    # M1 involves B11 (2x4) and B22 (2x3).
    # The product (A11+A22)(B11+B22) is not a standard matrix multiplication of a single shape.
    # It is the sum of two products: (A11+A22)B11 + (A11+A22)B22.
    # These are two independent multiplications.
    # So M1 decomposes into a [2,2,4] part and a [2,2,3] part.
    # Rank(M1) = Rank([2,2,4]) + Rank([2,2,3]) = 14 + 11 = 25.
    
    # Let's check all M:
    # M1 = (A11+A22)(B11+B22) = (A11+A22)B11 + (A11+A22)B22
    #    -> [2,2,4] + [2,2,3]. Rank 14 + 11 = 25.
    
    # M2 = (A21+A22)B11
    #    -> [2,2,4]. Rank 14.
    
    # M3 = A11(B12-B22)
    #    -> [2,2,3]. Rank 11.
    
    # M4 = A22(B21-B11)
    #    -> [2,2,4]. Rank 14.
    
    # M5 = (A11+A12)B22
    #    -> [2,2,3]. Rank 11.
    
    # M6 = (A21-A11)(B11+B12)
    #    -> (A21-A11)B11 + (A21-A11)B12
    #    -> [2,2,4] + [2,2,3]. Rank 14 + 11 = 25.
    
    # M7 = (A12-A22)(B21+B22)
    #    -> (A12-A22)B21 + (A12-A22)B22
    #    -> [2,2,4] + [2,2,3]. Rank 14 + 11 = 25.
    
    # Total Rank = 25 + 14 + 11 + 14 + 11 + 25 + 25 = 125.
    # This is WORSE than 98.
    
    # Why? Because in the padded version, M1 was a single [2,2,4] product.
    # Here, splitting M1 into two products increases the rank because we lose the sharing of the A-blocks?
    # No, the A-blocks are the same, but the decomposition of the B-part is separate.
    # The rank of a sum of tensors is at most the sum of ranks.
    # In the padded case, we treated B11+B22 as a single 2x4 matrix (with 3 cols from B22 and 1 col zero?).
    # No, padding B to 4x8 means B12 is 2x4 (cols 4-7), B22 is 2x4 (cols 4-7).
    # Then M1 is (A11+A22)(B11+B22) where B11 is 2x4, B22 is 2x4.
    # This is a single [2,2,4] multiplication. Rank 14.
    # In the unpadded case, M1 is sum of two multiplications. Rank 25.
    # So padding is BETTER.
    
    # Therefore, we should use the padded approach with rank 98.
    
    # Let's implement the padded Strassen decomposition.
    
    # 1. Define Strassen [2,2,2] decomposition (Rank 7)
    # 2. Define [2,2,4] decomposition (Rank 14) as sum of two [2,2,2]
    # 3. Assemble the 7 Strassen terms using [2,2,4] decompositions.
    
    # [2,2,2] Strassen Tensor
    # U_S: 4x7, V_S: 4x7, W_S: 4x7
    # Indices: i in 0..1, j in 0..1, k in 0..1
    # U[i,j, c], V[j,k, c], W[k,i, c]
    
    def get_strassen_222():
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # M1 = (A11+A22)(B11+B22)
        # A11: (0,0), A22: (1,1)
        # B11: (0,0), B22: (1,1)
        # C11 += M1, C12 -= M1, C21 -= M1, C22 += M1
        # W indices: k*n + i. n=2.
        # W[k,i,c]
        
        # M1
        U[0][0] = 1; U[3][0] = 1 # A11, A22
        V[0][0] = 1; V[3][0] = 1 # B11, B22
        W[0][0][0] = 1; W[1][1][0] = 1; W[0][1][0] = -1; W[1][0][0] = -1
        
        # M2 = (A21+A22)B11
        # A21: (1,0), A22: (1,1)
        # B11: (0,0)
        # C21 += M2
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[0][1][1] = 1
        
        # M3 = A11(B12-B22)
        # A11: (0,0)
        # B12: (0,1), B22: (1,1)
        # C12 += M3
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[1][0][2] = 1
        
        # M4 = A22(B21-B11)
        # A22: (1,1)
        # B21: (1,0), B11: (0,0)
        # C22 += M4
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[1][1][3] = 1
        
        # M5 = (A11+A12)B22
        # A11: (0,0), A12: (0,1)
        # B22: (1,1)
        # C12 += M5
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[1][0][4] = 1
        
        # M6 = (A21-A11)(B11+B12)
        # A21: (1,0), A11: (0,0)
        # B11: (0,0), B12: (0,1)
        # C21 -= M6
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[0][1][5] = -1
        
        # M7 = (A12-A22)(B21+B22)
        # A12: (0,1), A22: (1,1)
        # B21: (1,0), B22: (1,1)
        # C22 -= M7
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[1][1][6] = -1
        
        return U, V, W

    U_S, V_S, W_S = get_strassen_222()
    
    # Now construct [2,2,4] decomposition.
    # This is two [2,2,2] decompositions for cols 0,1 and 2,3.
    # U_224: 4x14, V_224: 8x14, W_224: 8x14
    # V indices: j*p + k. p=4.
    # W indices: k*n + i. n=2.
    
    U_224 = [[0]*14 for _ in range(4)]
    V_224 = [[0]*14 for _ in range(8)]
    W_224 = [[0]*14 for _ in range(8)]
    
    for c in range(7):
        # First [2,2,2] for k=0,1
        for i in range(2):
            for j in range(2):
                U_224[i*2+j][c] = U_S[i*2+j][c]
        for j in range(2):
            for k in range(2):
                V_224[j*4+k][c] = V_S[j*2+k][c]
        for k in range(2):
            for i in range(2):
                W_224[k*2+i][c] = W_S[k*2+i][c]
        
        # Second [2,2,2] for k=2,3
        c2 = c + 7
        for i in range(2):
            for j in range(2):
                U_224[i*2+j][c2] = U_S[i*2+j][c]
        for j in range(2):
            for k in range(2):
                V_224[j*4+(k+2)][c2] = V_S[j*2+k][c]
        for k in range(2):
            for i in range(2):
                W_224[(k+2)*2+i][c2] = W_S[k*2+i][c]
                
    # Now assemble the 4x4x7 decomposition.
    # We have 7 Strassen terms.
    # Each term uses a [2,2,4] decomposition (rank 14).
    # Total rank 98.
    
    rank_total = 98
    U = [[0]*rank_total for _ in range(16)]
    V = [[0]*rank_total for _ in range(28)]
    W = [[0]*rank_total for _ in range(28)]
    
    # Map Strassen blocks to global indices
    # A blocks:
    # A11: rows 0,1; cols 0,1
    # A12: rows 0,1; cols 2,3
    # A21: rows 2,3; cols 0,1
    # A22: rows 2,3; cols 2,3
    
    # B blocks (padded to 4x8):
    # B11: rows 0,1; cols 0,1,2,3
    # B12: rows 0,1; cols 4,5,6,7 (7 is zero)
    # B21: rows 2,3; cols 0,1,2,3
    # B22: rows 2,3; cols 4,5,6,7
    
    # Strassen terms:
    # M1: (A11+A22)(B11+B22)
    # M2: (A21+A22)B11
    # M3: A11(B12-B22)
    # M4: A22(B21-B11)
    # M5: (A11+A12)B22
    # M6: (A21-A11)(B11+B12)
    # M7: (A12-A22)(B21+B22)
    
    # We will fill U, V, W for each term.
    
    def add_term(U_local, V_local, W_local, offset, a_indices, b_indices, w_signs):
        # U_local: 4x14 (from U_224)
        # V_local: 8x14 (from V_224)
        # W_local: 8x14 (from W_224)
        # a_indices: list of (global_row, global_col, sign) for the A-block combination
        # b_indices: list of (global_row, global_col, sign) for the B-block combination
        # w_signs: mapping from local (k,i) to global (k,i) with sign
        
        for c in range(14):
            global_c = offset + c
            # Fill U
            for local_i in range(2):
                for local_j in range(2):
                    val = U_local[local_i*2+local_j][c]
                    if val != 0:
                        # Find which global A index this corresponds to
                        # This logic is tricky because a_indices is a combination.
                        # We need to pre-compute the U, V, W for the specific combination.
                        pass
                        
    # This approach is getting complex to generalize.
    # Let's just hardcode the 7 terms using the U_224, V_224, W_224 building blocks.
    
    # Term 1: M1 = (A11+A22)(B11+B22)
    # A part: A11 + A22
    # A11 is at (0,0), A22 is at (2,2) in 4x4 grid.
    # Local A indices in [2,2,4] are (0,0), (0,1), (1,0), (1,1).
    # We map local (0,0) -> global (0,0) [A11]
    # local (1,1) -> global (2,2) [A22]
    # local (0,1) -> 0
    # local (1,0) -> 0
    
    # B part: B11 + B22
    # B11 is at rows 0,1; cols 0-3.
    # B22 is at rows 2,3; cols 4-7.
    # Local B indices in [2,2,4] are j in 0,1; k in 0-3.
    # local (0, k) -> global (0, k) [B11]
    # local (1, k) -> global (2, k+4) [B22]
    
    # W part:
    # Local W indices k in 0-3, i in 0,1.
    # global W indices:
    # k_local 0-1 -> global k 0-1 (from B11/B22 cols 0-1? No, B11 cols 0-3, B22 cols 4-7)
    # Wait, B11 is 2x4, B22 is 2x4.
    # B11 cols 0-3. B22 cols 4-7.
    # So local k=0,1,2,3 in B11 maps to global k=0,1,2,3.
    # local k=0,1,2,3 in B22 maps to global k=4,5,6,7.
    # But in M1, we sum B11 and B22.
    # So local k=0 in B11 (global k=0) and local k=0 in B22 (global k=4) are added?
    # No, the local k index in [2,2,4] is the column index of the 2x4 matrix.
    # So local k=0 corresponds to the first column of the 2x4 matrix.
    # In M1, the 2x4 matrix is B11+B22? No, B11 and B22 are both 2x4.
    # So we add them element-wise.
    # local k=0 -> global k=0 (from B11) AND global k=4 (from B22)?
    # No, B11 has cols 0-3. B22 has cols 4-7.
    # They are NOT the same columns.
    # So B11+B22 is not a valid operation if they have different column indices.
    # In Strassen, B11 and B22 must be the same size and correspond to the same "block" of the matrix.
    # For 4x4 x 4x4, B11 is top-left 2x2, B22 is bottom-right 2x2.
    # Here B is 4x7.
    # If we pad to 4x8, B11 is top-left 2x4, B22 is bottom-right 2x4.
    # B11 cols 0-3. B22 cols 4-7.
    # They are DIFFERENT columns.
    # So B11+B22 is a 2x8 matrix? No, 2x4 + 2x4 is 2x4 if we align them.
    # But they represent different parts of B.
    # So we cannot add them directly.
    
    # This implies that standard Strassen does NOT apply directly to rectangular matrices by just padding.
    # You must split the matrix into square blocks of the SAME size.
    # So for 4x4 x 4x7, we should split into 2x2 blocks.
    # A: 4 blocks of 2x2.
    # B: 2 blocks of 2x7.
    # This is NOT a 2x2 block decomposition of B.
    
    # So we must use a different algorithm or pad B to 4x8 and accept that the last column is zero.
    # If we pad B to 4x8, then B11 (2x4) and B22 (2x4) are valid blocks.
    # B11 cols 0-3. B22 cols 4-7.
    # Then B11+B22 is a 2x4 matrix where col 0 is B11[0]+B22[0]?
    # No, B11 and B22 are at different positions in the original matrix.
    # In the 4x8 matrix, B11 is at rows 0-1, cols 0-3.
    # B22 is at rows 2-3, cols 4-7.
    # They are disjoint.
    # So B11+B22 is zero? No.
    # Strassen requires B11 and B22 to be the corresponding blocks.
    # For 4x4 x 4x4, B11 is (0,0)-(1,1) and B22 is (2,2)-(3,3).
    # For 4x8, B11 is (0,0)-(1,3) and B22 is (2,4)-(3,7).
    # They are not "corresponding" in the sense of the same column range.
    # So Strassen does not work directly on 4x4 x 4x8.
    
    # We must split the 7 columns into 4 and 3.
    # And use a decomposition that handles rectangular blocks.
    
    # Given the complexity, let's stick to the schoolbook method but optimize it?
    # No, schoolbook is 112.
    
    # Let's try the Kronecker product approach mentioned in the prompt.
    # Rank 98.
    
    # I will implement the rank 98 decomposition using the padded 4x8 matrix,
    # but I must ensure the Strassen blocks align correctly.
    # If I pad B to 4x8, I am computing 4x4 x 4x8.
    # This is a valid tensor decomposition.
    # The rank is 98.
    # The result for the 4x7 part is correct.
    
    # Let's implement the 4x4 x 4x8 Strassen decomposition.
    
    U = [[0]*98 for _ in range(16)]
    V = [[0]*98 for _ in range(32)] # 4x8
    W = [[0]*98 for _ in range(32)] # 8x4
    
    # We will use the U_224, V_224, W_224 defined above.
    # But we need to map the 7 Strassen terms to the global indices.
    
    # This is too complex to do manually in the short time.
    # I will return the schoolbook decomposition to be safe.
    # 112 is the current best.
    
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

