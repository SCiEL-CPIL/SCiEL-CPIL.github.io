DIMENSIONS = [5, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # Rank 49 for 4x4 core
    # Rank 45 for the rest
    rank = 94
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # --- Part 1: 4x4 core using Kronecker product of Strassen 2x2 (Rank 7) ---
    # Strassen 2x2 decomposition coefficients
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # We need to build the U, V, W matrices for the 4x4 multiplication.
    # The indices for 4x4 are i,j in 0..3.
    # We map the 4x4 indices to the 5x5 indices 0..3.
    
    # Strassen 2x2 U, V, W vectors (for i,j in 0..1)
    # U[i, k], V[j, k], W[k, l] where k is the rank index 0..6.
    # For 2x2:
    # U[0,0]=1, U[1,0]=1, U[2,0]=1, U[3,0]=1  (A11+A22) -> Wait, U is for A.
    # A = [[A11, A12], [A21, A22]]
    # U[0,0]=1, U[1,0]=1 (A11+A22 is M1, M5? No)
    
    # Let's define the 7 triples for 2x2.
    # (u, v, w) where u is A index, v is B index, w is C index.
    # Strassen:
    # 1: (A11+A22, B11+B22, C11+... )
    # Actually, it's easier to just generate the Kronecker product.
    
    strassen_u = [
        [1, 1, 0, 0, 1, 0, 0], # A11
        [0, 1, 1, 0, 0, 1, -1], # A21
        [1, 0, 0, 0, 1, -1, 1], # A12
        [0, 0, 0, 1, 0, 0, -1]  # A22
    ]
    # Wait, the standard Strassen matrices are:
    # U1 = [[1,1,0,0],[1,0,1,-1],[0,1,0,0],[1,0,0,1]]? 
    # Let's use the explicit linear forms.
    # M1: A11+A22, B11+B22
    # M2: A21+A22, B11
    # M3: A11, B12-B22
    # M4: A22, B21-B11
    # M5: A11+A12, B22
    # M6: A21-A11, B11+B12
    # M7: A12-A22, B21+B22
    
    # U (A) coefficients for M1..M7:
    # M1: A11(1), A22(1) -> [1, 0, 0, 1] (row 0, 1, 2, 3)
    # M2: A21(1), A22(1) -> [0, 1, 0, 1]
    # M3: A11(1) -> [1, 0, 0, 0]
    # M4: A22(1) -> [0, 0, 0, 1]
    # M5: A11(1), A12(1) -> [1, 0, 1, 0]
    # M6: A21(1), A11(-1) -> [-1, 1, 0, 0]
    # M7: A12(1), A22(-1) -> [0, 0, 1, -1]
    
    # V (B) coefficients for M1..M7:
    # M1: B11(1), B22(1) -> [1, 0, 0, 1]
    # M2: B11(1) -> [1, 0, 0, 0]
    # M3: B12(1), B22(-1) -> [0, 1, 0, -1]
    # M4: B21(1), B11(-1) -> [-1, 0, 1, 0]
    # M5: B22(1) -> [0, 0, 0, 1]
    # M6: B11(1), B12(1) -> [1, 1, 0, 0]
    # M7: B21(1), B22(1) -> [0, 0, 1, 1]
    
    # W (C) coefficients for M1..M7:
    # C11 = M1 + M4 - M5 + M7 -> M1(1), M4(1), M5(-1), M7(1)
    # C12 = M3 + M5 -> M3(1), M5(1)
    # C21 = M2 + M4 -> M2(1), M4(1)
    # C22 = M1 - M2 + M3 + M6 -> M1(1), M2(-1), M3(1), M6(1)
    
    # W is (p*n, rank) = (4, 7) for 2x2.
    # W[0, 0]=1, W[0, 1]=1, W[0, 2]=-1, W[0, 3]=1
    # W[1, 2]=1, W[1, 3]=1
    # W[2, 1]=1, W[2, 3]=1
    # W[3, 0]=1, W[3, 1]=-1, W[3, 2]=1, W[3, 5]=1
    
    # Now Kronecker Product for 4x4.
    # The rank is 7*7=49.
    # The indices for 4x4 A are 0..15.
    # A_4x4[i, j] = sum_{k} U4[i, k] * V4[j, k] * W4[k, l]
    
    # We construct U4, V4, W4.
    U4 = [[0]*49 for _ in range(16)]
    V4 = [[0]*49 for _ in range(16)]
    W4 = [[0]*49 for _ in range(16)]
    
    for k1 in range(7):
        for k2 in range(7):
            k4 = k1 * 7 + k2
            for i1 in range(4):
                for i2 in range(4):
                    i4 = i1 * 4 + i2
                    U4[i4][k4] = strassen_u[i1][k1] * strassen_u[i2][k2]
                    
            for j1 in range(4):
                for j2 in range(4):
                    j4 = j1 * 4 + j2
                    V4[j4][k4] = strassen_v[j1][k1] * strassen_v[j2][k2]
                    
            for l1 in range(4):
                for l2 in range(4):
                    l4 = l1 * 4 + l2
                    W4[l4][k4] = strassen_w[l1][k1] * strassen_w[l2][k2]
                    
    # Copy U4, V4, W4 to U, V, W for indices 0..3
    for i in range(16):
        for k in range(49):
            U[i][k] = U4[i][k]
            V[i][k] = V4[i][k]
            W[i][k] = W4[i][k]
            
    # --- Part 2: 5th row and column ---
    # Rank offset = 49
    # We need to compute:
    # C[i, 4] for i in 0..3: A[i, 0..3]B[0..3, 4] + A[i, 4]B[4, 4]
    # C[4, j] for j in 0..3: A[4, 0..3]B[0..3, j] + A[4, 4]B[4, j]
    # C[4, 4]: A[4, 0..3]B[0..3, 4] + A[4, 4]B[4, 4]
    
    # We will use schoolbook for these.
    # C[i, 4] = sum_{j=0}^3 A[i, j] B[j, 4] + A[i, 4] B[4, 4]
    # This is 4 terms for the sum + 1 term for the diagonal.
    # Total 5 terms per i?
    # Actually, we can just add 5 terms for each C[i, 4] and C[4, j].
    # And 5 terms for C[4, 4].
    # Total 5*4 + 5*4 + 5 = 45 terms.
    
    col = 49
    # C[i, 4] for i=0..3
    for i in range(4):
        for j in range(4):
            U[i * 5 + j][col] = 1
            V[j * 5 + 4][col] = 1
            W[4 * 5 + i][col] = 1
            col += 1
        # Term A[i, 4] B[4, 4]
        U[i * 5 + 4][col] = 1
        V[4 * 5 + 4][col] = 1
        W[4 * 5 + i][col] = 1
        col += 1
        
    # C[4, j] for j=0..3
    for j in range(4):
        for i in range(4):
            U[4 * 5 + i][col] = 1
            V[i * 5 + j][col] = 1
            W[4 * 5 + j][col] = 1
            col += 1
        # Term A[4, 4] B[4, j]
        U[4 * 5 + 4][col] = 1
        V[4 * 5 + j][col] = 1
        W[4 * 5 + j][col] = 1
        col += 1
        
    # C[4, 4]
    for i in range(4):
        U[4 * 5 + i][col] = 1
        V[i * 5 + 4][col] = 1
        W[4 * 5 + 4][col] = 1
        col += 1
    # Term A[4, 4] B[4, 4]
    U[4 * 5 + 4][col] = 1
    V[4 * 5 + 4][col] = 1
    W[4 * 5 + 4][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}

# Note: strassen_u, v, w definitions needed
strassen_u = [
    [1, 0, 1, 0, 1, -1, 0], # A11
    [0, 1, 0, 0, 0, 1, 0],  # A21
    [0, 0, 0, 0, 1, 0, 1],  # A12
    [1, 1, 0, 1, 0, 0, -1]  # A22
]
strassen_v = [
    [1, 1, 0, -1, 0, 1, 0], # B11
    [0, 0, 1, 0, 0, 1, 0],  # B12
    [0, 0, 0, 1, 0, 0, 1],  # B21
    [1, 0, -1, 0, 1, 0, 1]  # B22
]
strassen_w = [
    [1, 0, 0, 1, 0, 0, 1], # C11
    [0, 0, 1, 0, 1, 0, 0], # C12
    [0, 1, 0, 1, 0, 0, 0], # C21
    [1, -1, 1, 0, 0, 1, 0] # C22
]
