"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    def schoolbook(n, m, p):
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W

    def strassen_222():
        # Rank 7 decomposition for 2x2x2
        # Indices: A(i,j), B(j,k), C(i,k) for i,j,k in 0..1
        # U shape (4, 7), V shape (4, 7), W shape (4, 7)
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # M1 = (A00+A11)(B00+B11)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        W[1][0] = -1; W[2][0] = -1
        
        # M2 = (A10+A11)B00
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1; W[3][1] = 1
        W[1][1] = -1
        
        # M3 = A00(B01-B11)
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[0][2] = 1; W[1][2] = 1
        
        # M4 = A11(B10-B00)
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[2][3] = 1; W[3][3] = 1
        
        # M5 = (A00+A01)B11
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = 1; W[1][4] = 1
        W[2][4] = -1
        
        # M6 = (A10-A00)(B00+B01)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[2][5] = 1; W[3][5] = 1
        W[0][5] = -1
        
        # M7 = (A01-A11)(B10+B11)
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[1][6] = 1; W[3][6] = 1
        W[2][6] = -1
        
        return U, V, W

    def construct_333_rank26():
        # Decomposition of 3x3x3 into blocks:
        # 1. A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2] (Rank 7)
        # 2. A[0:2, 0:2] * B[0:2, 2]   -> C[0:2, 2]   (Rank 4)
        # 3. A[0:2, 2]   * B[2, 0:3]   -> C[0:2, 0:3] (Rank 6)
        # 4. A[2, 0:2]   * B[0:2, 0:3] -> C[2, 0:3]   (Rank 6)
        # 5. A[2, 2]     * B[2, 0:3]   -> C[2, 0:3]   (Rank 3)
        
        U = [[0]*26 for _ in range(9)]
        V = [[0]*26 for _ in range(9)]
        W = [[0]*26 for _ in range(9)]
        
        # Part 1: Strassen for top-left 2x2
        U1, V1, W1 = strassen_222()
        for r in range(7):
            for i in range(4):
                U[i][r] += U1[i][r]
                V[i][r] += V1[i][r]
                W[i][r] += W1[i][r]
        
        # Part 2: A[0:2, 0:2] * B[0:2, 2] -> C[0:2, 2]
        # A indices: 0,1,2,3 (0:2, 0:2)
        # B indices: 4,5 (0:2, 2) -> B[j,2] for j=0,1. V index = j*3 + 2. j=0->2, j=1->5.
        # C indices: 2, 5 (0:2, 2) -> C[i,2] for i=0,1. W index = k*3 + i. k=2, i=0->6, i=1->7.
        col = 7
        for i in range(2):
            for j in range(2):
                U[i*3 + j][col] = 1
                V[j*3 + 2][col] = 1
                W[2*3 + i][col] = 1
                col += 1
                # col is now 11
                
        # Part 3: A[0:2, 2] * B[2, 0:3] -> C[0:2, 0:3]
        # A indices: 2, 5 (0:2, 2) -> A[i,2] for i=0,1. U index = i*3 + 2. i=0->2, i=1->5.
        # B indices: 6,7,8 (2, 0:3) -> B[2,k] for k=0,1,2. V index = 2*3 + k. k=0->6, k=1->7, k=2->8.
        # C indices: 0..5 (0:2, 0:3) -> C[i,k] for i=0,1, k=0,1,2. W index = k*3 + i.
        for i in range(2):
            for k in range(3):
                U[i*3 + 2][col] = 1
                V[2*3 + k][col] = 1
                W[k*3 + i][col] = 1
                col += 1
                # col is now 17

        # Part 4: A[2, 0:2] * B[0:2, 0:3] -> C[2, 0:3]
        # A indices: 6, 7 (2, 0:2) -> A[2,j] for j=0,1. U index = 2*3 + j. j=0->6, j=1->7.
        # B indices: 0..5 (0:2, 0:3) -> B[j,k] for j=0,1, k=0,1,2. V index = j*3 + k.
        # C indices: 6, 7, 8 (2, 0:3) -> C[2,k] for k=0,1,2. W index = k*3 + 2.
        for j in range(2):
            for k in range(3):
                U[2*3 + j][col] = 1
                V[j*3 + k][col] = 1
                W[k*3 + 2][col] = 1
                col += 1
                # col is now 23

        # Part 5: A[2, 2] * B[2, 0:3] -> C[2, 0:3]
        # A index: 8 (2, 2)
        # B indices: 6, 7, 8 (2, 0:3)
        # C indices: 6, 7, 8 (2, 0:3)
        for k in range(3):
            U[8][col] = 1
            V[2*3 + k][col] = 1
            W[k*3 + 2][col] = 1
            col += 1
            # col is now 26

        return U, V, W

    def construct_331_rank9():
        # Schoolbook for 3x3x1
        return schoolbook(3, 3, 1)

    # Combine
    U1, V1, W1 = construct_333_rank26()
    U2, V2, W2 = construct_333_rank26()
    U3, V3, W3 = construct_331_rank9()
    
    U = [u1 + u2 + u3 for u1, u2, u3 in zip(U1, U2, U3)]
    V = [v1 + v2 + v3 for v1, v2, v3 in zip(V1, V2, V3)]
    W = [w1 + w2 + w3 for w1, w2, w3 in zip(W1, W2, W3)]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
