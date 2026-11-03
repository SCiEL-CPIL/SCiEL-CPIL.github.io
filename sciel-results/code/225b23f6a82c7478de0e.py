import time

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # We use a block Strassen decomposition for the 4x4 part (rank 49)
    # and add 4 terms for the 5th column (rank 4).
    # Total rank 53.
    
    # Strassen 2x2 factors
    # U, V, W are 2x2 matrices for each of the 7 terms.
    # Indices: U[i, j], V[j, k], W[k, i]
    
    # Term 1: (A00+A11)(B00+B11)
    # U: diag(1, 1) -> A00, A11
    # V: diag(1, 1) -> B00, B11
    # W: all 1s
    U1 = [[1, 0], [0, 1]]
    V1 = [[1, 0], [0, 1]]
    W1 = [[1, 1], [1, 1]] # W[k, i] -> W[0,0]=1, W[0,1]=1, W[1,0]=1, W[1,1]=1
    
    # Term 2: (A10+A11)B00
    # U: row 1 is [1, 1], row 0 is [0, 0]
    # V: col 0 is [1, 0], col 1 is [0, 0]
    # W: C10 gets M2, C11 gets M2
    U2 = [[0, 0], [1, 1]]
    V2 = [[1, 0], [0, 0]]
    W2 = [[0, 0], [1, 1]] # W[1,0]=1, W[1,1]=1
    
    # Term 3: A00(B10-B11)
    # U: row 0 is [1, 0], row 1 is [0, 0]
    # V: col 0 is [0, 1], col 1 is [0, -1]
    # W: C01 gets M3, C10 gets M3
    U3 = [[1, 0], [0, 0]]
    V3 = [[0, 0], [1, -1]]
    W3 = [[0, 1], [1, 0]] # W[1,0]=1 (C10), W[0,1]=1 (C01)
    
    # Term 4: A11(B01-B00)
    # U: row 1 is [0, 1], row 0 is [0, 0]
    # V: col 0 is [0, -1], col 1 is [0, 1]
    # W: C00 gets M4, C11 gets M4
    U4 = [[0, 0], [0, 1]]
    V4 = [[0, 0], [-1, 1]]
    W4 = [[1, 0], [0, 1]] # W[0,0]=1 (C00), W[1,1]=1 (C11)
    
    # Term 5: (A00+A01)B11
    # U: row 0 is [1, 1], row 1 is [0, 0]
    # V: col 1 is [0, 1], col 0 is [0, 0]
    # W: C00 gets -M5, C10 gets M5
    U5 = [[1, 1], [0, 0]]
    V5 = [[0, 0], [0, 1]]
    W5 = [[-1, 0], [1, 0]] # W[0,0]=-1 (C00), W[1,0]=1 (C10)
    
    # Term 6: (A10-A00)(B00+B01)
    # U: row 1 is [-1, 0], row 0 is [0, 0]? No.
    # A10 - A00.
    # U: row 0 is [-1, 0], row 1 is [1, 0]
    # V: col 0 is [1, 1], col 1 is [0, 0]
    # W: C01 gets M6, C10 gets -M6
    U6 = [[-1, 0], [1, 0]]
    V6 = [[1, 0], [1, 0]]
    W6 = [[0, 1], [-1, 0]] # W[1,0]=-1 (C10), W[0,1]=1 (C01)
    
    # Term 7: (A01-A11)(B10+B11)
    # U: row 0 is [0, 1], row 1 is [0, -1]
    # V: col 0 is [0, 1], col 1 is [0, 1]
    # W: C00 gets M7, C11 gets M7
    U7 = [[0, 1], [0, -1]]
    V7 = [[0, 0], [1, 1]]
    W7 = [[1, 0], [0, 1]] # W[0,0]=1 (C00), W[1,1]=1 (C11)
    
    strassen_terms = [
        (U1, V1, W1),
        (U2, V2, W2),
        (U3, V3, W3),
        (U4, V4, W4),
        (U5, V5, W5),
        (U6, V6, W6),
        (U7, V7, W7)
    ]
    
    # We need to map 2x2 blocks to 4x4 indices.
    # A is 4x4. Blocks: A00 (0:2, 0:2), A01 (0:2, 2:4), A10 (2:4, 0:2), A11 (2:4, 2:4).
    # B is 4x5. Blocks: B00 (0:2, 0:2), B01 (0:2, 2:4), B10 (2:4, 0:2), B11 (2:4, 2:4).
    # Note: B has 5 columns. B01 is 2x2, B11 is 2x2.
    # The 5th column is handled separately.
    
    rank = 0
    U = []
    V = []
    W = []
    
    # 49 terms for the 4x4 part
    for k in range(4): # Block row of A? No, we iterate over the 4 block positions of the output C?
        # Actually, the Strassen algorithm computes the whole 2x2 block matrix C.
        # So we just iterate over the 7 Strassen terms.
        # Each term corresponds to a product of two 2x2 block matrices.
        # The U, V, W factors for the 2x2 case are given.
        # We need to expand them to 4x4.
        
        for t in range(7):
            Ut, Vt, Wt = strassen_terms[t]
            
            # Ut is 2x2. It defines the linear combination of A's blocks.
            # Vt is 2x2. It defines the linear combination of B's blocks.
            # Wt is 2x2. It defines the contribution to C's blocks.
            
            # We need to create one factor for each element of the 2x2 block matrices?
            # No, the Strassen decomposition is for the 2x2 tensor.
            # The rank is 7.
            # For the 4x4 case, we have 4 "2x2" sub-problems?
            # No, the block Strassen computes the 4x4 product using 7 "2x2" products.
            # Each "2x2" product has rank 7.
            # So total rank 49.
            
            # The factors for the 4x4 case are constructed by placing the 2x2 factors
            # into the appropriate 4x4 slots.
            
            # U_4x4 is 4x4.
            # For a given Strassen term t, U_4x4 is formed by placing Ut into the block structure.
            # But Ut is a 2x2 matrix of coefficients for the blocks A00, A01, A10, A11.
            # So U_4x4[i, j] = Ut[i//2, j//2].
            
            U4 = [[0]*4 for _ in range(4)]
            for i in range(2):
                for j in range(2):
                    U4[i*2][j*2] = Ut[i][j]
                    U4[i*2][j*2+1] = Ut[i][j]
                    U4[i*2+1][j*2] = Ut[i][j]
                    U4[i*2+1][j*2+1] = Ut[i][j]
            
            V4 = [[0]*4 for _ in range(4)] # V is 4x5? No, V is m x p = 4x5.
            # But the 4x4 part only uses the first 4 columns of B.
            # So V4 is 4x4 effectively, but we need 4x5.
            for i in range(2):
                for j in range(2):
                    V4[i*2][j*2] = Vt[i][j]
                    V4[i*2][j*2+1] = Vt[i][j]
                    V4[i*2+1][j*2] = Vt[i][j]
                    V4[i*2+1][j*2+1] = Vt[i][j]
            # The 5th column of V4 is 0.
            
            W4 = [[0]*4 for _ in range(4)] # W is p x n = 5x4.
            # W4 is 4x4 effectively (first 4 rows).
            for i in range(2):
                for j in range(2):
                    W4[i*2][j*2] = Wt[i][j]
                    W4[i*2][j*2+1] = Wt[i][j]
                    W4[i*2+1][j*2] = Wt[i][j]
                    W4[i*2+1][j*2+1] = Wt[i][j]
            
            # Add to global U, V, W
            U.append(U4)
            V.append(V4 + [[0]*4 for _ in range(4)] + [0]) # Wait, V is 4x5.
            # V4 is 4x4. We need to pad it to 4x5.
            V_padded = [row + [0] for row in V4]
            V.append(V_padded)
            
            W_padded = W4 + [[0]*4] # W is 5x4. Pad 5th row.
            W.append(W_padded)
            
            rank += 1
            
    # Add 4 terms for the 5th column
    # C[i, 4] = sum_j A[i, j] B[j, 4]
    for j in range(4):
        Uj = [[0]*4 for _ in range(4)]
        Uj[j][j] = 1 # Wait, U is n x m = 4x4.
        # U[i, j] coefficient for A[i, j].
        # We want sum_j U[i, j] A[i, j] = A[i, j0].
        # So U[i, j] = delta(i, j0)? No.
        # U is a matrix. U[i, j].
        # We want the product to be A[i, j0].
        # So U[i, j] should be 1 if j == j0, else 0.
        # And it should be independent of i?
        # No, U is a factor.
        # For term j0:
        # U[i, j] = delta(j, j0).
        for i in range(4):
            Uj[i][j] = 1
            
        Vj = [[0]*5 for _ in range(4)]
        # V[j, k] coefficient for B[j, k].
        # We want sum_k V[j, k] B[j, k] = B[j0, 4].
        # So V[j, k] = 1 if j == j0 and k == 4.
        Vj[j][4] = 1
        
        Wj = [[0]*4 for _ in range(5)]
        # W[k, i] coefficient for C[k, i].
        # We want C[i, 4] to get the product.
        # So W[4, i] = 1 for all i.
        for i in range(4):
            Wj[4][i] = 1
            
        U.append(Uj)
        V.append(Vj)
        W.append(Wj)
        rank += 1
        
    return {"U": U, "V": V, "W": W}
