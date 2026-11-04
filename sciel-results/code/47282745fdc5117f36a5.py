"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS

    # We will build a decomposition using block decomposition with Strassen
    # for 2x2x2 sub-blocks.
    
    # The idea: we split the middle dimension m=4 into 2+2.
    # Then we have two sub-problems of size 3x2x5.
    # For each 3x2x5, we split n=3 into 1+2.
    # Sub-sub-problem 1: 1x2x5 (rank 10, schoolbook)
    # Sub-sub-problem 2: 2x2x5
    # For 2x2x5, split p=5 into 2+3.
    # Sub-sub-sub-problem 1: 2x2x2 (Strassen, rank 7)
    # Sub-sub-sub-problem 2: 2x2x3 (schoolbook, rank 12)
    # Total for 2x2x5: 7+12 = 19
    # Total for 3x2x5: 10+19 = 29
    # Total for 3x4x5: 29+29 = 58

    # We'll build this recursively.
    
    # Helper: schoolbook decomposition for n x m x p
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
        return U, V, W, rank

    # Helper: Strassen decomposition for 2x2x2
    def strassen_222():
        # 7 terms for 2x2x2 matrix multiplication
        # C = A*B where A is 2x2, B is 2x2, C is 2x2
        # U is 2x2 (n x m), V is 2x2 (m x p), W is 2x2 (p x n)
        # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
        # 
        # Strassen's 7 products:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        #
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        #
        # In tensor decomposition form, each "product" corresponds to one rank-1 term.
        # For rank r term, we have U[:,r], V[:,r], W[:,r]
        # U[i*2+j, r] * V[j*2+k, r] * W[k*2+i, r] = C[i,k]
        
        U = [[0] * 7 for _ in range(4)]  # 2x2 = 4 rows
        V = [[0] * 7 for _ in range(4)]  # 2x2 = 4 rows
        W = [[0] * 7 for _ in range(4)]  # 2x2 = 4 rows
        
        # M1 = (A11+A22)(B11+B22)
        # U: A11=1, A22=1 (rows 0 and 3 in U)
        # V: B11=1, B22=1 (rows 0 and 3 in V)
        # W: C11=1, C22=1 (rows 0 and 3 in W)
        # W[k*2+i]: C11 -> k=0,i=0 -> row 0; C22 -> k=1,i=1 -> row 3
        U[0][0] = 1  # A11
        U[3][0] = 1  # A22
        V[0][0] = 1  # B11
        V[3][0] = 1  # B22
        W[0][0] = 1  # C11
        W[3][0] = 1  # C22
        
        # M2 = (A21+A22)B11
        # U: A21=1 (row 2), A22=1 (row 3)
        # V: B11=1 (row 0)
        # W: C21=1 (k=1,i=0 -> row 2)
        U[2][1] = 1  # A21
        U[3][1] = 1  # A22
        V[0][1] = 1  # B11
        W[2][1] = 1  # C21
        
        # M3 = A11(B21-B22)
        # U: A11=1 (row 0)
        # V: B21=1 (row 2), B22=-1 (row 3)
        # W: C12=1 (k=0,i=1 -> row 1)
        U[0][2] = 1  # A11
        V[2][2] = 1  # B21
        V[3][2] = -1  # B22
        W[1][2] = 1  # C12
        
        # M4 = A22(B12-B11)
        # U: A22=1 (row 3)
        # V: B12=1 (row 1), B11=-1 (row 0)
        # W: C11=1 (row 0), C21=1 (row 2)
        U[3][3] = 1  # A22
        V[1][3] = 1  # B12
        V[0][3] = -1  # B11
        W[0][3] = 1  # C11
        W[2][3] = 1  # C21
        
        # M5 = (A11+A12)B22
        # U: A11=1 (row 0), A12=1 (row 1)
        # V: B22=1 (row 3)
        # W: C11=-1 (row 0), C12=1 (row 1)
        U[0][4] = 1  # A11
        U[1][4] = 1  # A12
        V[3][4] = 1  # B22
        W[0][4] = -1  # C11
        W[1][4] = 1  # C12
        
        # M6 = (A21-A11)(B11+B12)
        # U: A21=1 (row 2), A11=-1 (row 0)
        # V: B11=1 (row 0), B12=1 (row 1)
        # W: C22=1 (row 3)
        U[2][5] = 1  # A21
        U[0][5] = -1  # A11
        V[0][5] = 1  # B11
        V[1][5] = 1  # B12
        W[3][5] = 1  # C22
        
        # M7 = (A12-A22)(B21+B22)
        # U: A12=1 (row 1), A22=-1 (row 3)
        # V: B21=1 (row 2), B22=1 (row 3)
        # W: C11=1 (row 0)
        U[1][6] = 1  # A12
        U[3][6] = -1  # A22
        V[2][6] = 1  # B21
        V[3][6] = 1  # B22
        W[0][6] = 1  # C11
        
        return U, V, W, 7

    # Helper: combine two decompositions along the m (middle) dimension
    def combine_along_m(U1, V1, W1, r1, U2, V2, W2, r2, n, m1, m2, p):
        # U1 is n*m1 x r1, V1 is m1*p x r1, W1 is p*n x r1
        # U2 is n*m2 x r2, V2 is m2*p x r2, W2 is p*n x r2
        # Result: U is n*(m1+m2) x (r1+r2), V is (m1+m2)*p x (r1+r2), W is p*n x (r1+r2)
        
        m = m1 + m2
        r = r1 + r2
        
        U = [[0] * r for _ in range(n * m)]
        V = [[0] * r for _ in range(m * p)]
        W = [[0] * r for _ in range(p * n)]
        
        # Fill first r1 columns
        for i in range(n):
            for j in range(m1):
                for c in range(r1):
                    U[i * m + j][c] = U1[i * m1 + j][c]
        for j in range(m1):
            for k in range(p):
                for c in range(r1):
                    V[j * p + k][c] = V1[j * p + k][c]
        for k in range(p):
            for i in range(n):
                for c in range(r1):
                    W[k * n + i][c] = W1[k * n + i][c]
        
        # Fill next r2 columns (offset by r1)
        for i in range(n):
            for j in range(m2):
                for c in range(r2):
                    U[i * m + m1 + j][r1 + c] = U2[i * m2 + j][c]
        for j in range(m2):
            for k in range(p):
                for c in range(r2):
                    V[(m1 + j) * p + k][r1 + c] = V2[j * p + k][c]
        for k in range(p):
            for i in range(n):
                for c in range(r2):
                    W[k * n + i][r1 + c] = W2[k * n + i][c]
        
        return U, V, W, r

    # Helper: combine two decompositions along the n (first) dimension
    def combine_along_n(U1, V1, W1, r1, n1, U2, V2, W2, r2, m, p):
        # U1 is n1*m x r1, V1 is m*p x r1, W1 is p*n1 x r1
        # U2 is n2*m x r2, V2 is m*p x r2, W2 is p*n2 x r2
        # Result: U is (n1+n2)*m x (r1+r2), V is m*p x (r1+r2), W is p*(n1+n2) x (r1+r2)
        
        n = n1 + len(U2) // m  # n2 = len(U2) // m
        r = r1 + r2
        
        U = [[0] * r for _ in range(n * m)]
        V = [[0] * r for _ in range(m * p)]
        W = [[0] * r for _ in range(p * n)]
        
        # Fill first r1 columns
        for i in range(n1):
            for j in range(m):
                for c in range(r1):
                    U[i * m + j][c] = U1[i * m + j][c]
        for j in range(m):
            for k in range(p):
                for c in range(r1):
                    V[j * p + k][c] = V1[j * p + k][c]
        for k in range(p):
            for i in range(n1):
                for c in range(r1):
                    W[k * n + i][c] = W1[k * n1 + i][c]
        
        # Fill next r2 columns (offset by r1)
        n2 = len(U2) // m
        for i in range(n2):
            for j in range(m):
                for c in range(r2):
                    U[(n1 + i) * m + j][r1 + c] = U2[i * m + j][c]
        for j in range(m):
            for k in range(p):
                for c in range(r2):
                    V[j * p + k][r1 + c] = V2[j * p + k][c]
        for k in range(p):
            for i in range(n2):
                for c in range(r2):
                    W[k * n + n1 + i][r1 + c] = W2[k * n2 + i][c]
        
        return U, V, W, r

    # Helper: combine two decompositions along the p (last) dimension
    def combine_along_p(U1, V1, W1, r1, n, m, p1, U2, V2, W2, r2, p2):
        # U1 is n*m x r1, V1 is m*p1 x r1, W1 is p1*n x r1
        # U2 is n*m x r2, V2 is m*p2 x r2, W2 is p2*n x r2
        # Result: U is n*m x (r1+r2), V is m*(p1+p2) x (r1+r2), W is (p1+p2)*n x (r1+r2)
        
        p = p1 + p2
        r = r1 + r2
        
        U = [[0] * r for _ in range(n * m)]
        V = [[0] * r for _ in range(m * p)]
        W = [[0] * r for _ in range(p * n)]
        
        # Fill first r1 columns
        for i in range(n):
            for j in range(m):
                for c in range(r1):
                    U[i * m + j][c] = U1[i * m + j][c]
        for j in range(m):
            for k in range(p1):
                for c in range(r1):
                    V[j * p + k][c] = V1[j * p1 + k][c]
        for k in range(p1):
            for i in range(n):
                for c in range(r1):
                    W[k * n + i][c] = W1[k * n1 + i][c] if False else W1[k * n + i][c]
        
        # Fix W1 indexing: W1 is p1*n x r1, so W1[k*n + i] where k in [0,p1), i in [0,n)
        # But in the full W, index is k*n + i where k in [0,p), i in [0,n)
        # So for first part, k in [0,p1), offset is 0
        # For second part, k in [0,p2), offset is p1
        
        # Let me redo this more carefully
        U = [[0] * r for _ in range(n * m)]
        V = [[0] * r for _ in range(m * p)]
        W = [[0] * r for _ in range(p * n)]
        
        # First r1 columns
        for row in range(n * m):
            for c in range(r1):
                U[row][c] = U1[row][c]
        for j in range(m):
            for k in range(p1):
                for c in range(r1):
                    V[j * p + k][c] = V1[j * p1 + k][c]
        for k in range(p1):
            for i in range(n):
                for c in range(r1):
                    W[k * n + i][c] = W1[k * n + i][c]
        
        # Next r2 columns (offset by r1)
        for row in range(n * m):
            for c in range(r2):
                U[row][r1 + c] = U2[row][c]
        for j in range(m):
            for k in range(p2):
                for c in range(r2):
                    V[j * p + p1 + k][r1 + c] = V2[j * p2 + k][c]
        for k in range(p2):
            for i in range(n):
                for c in range(r2):
                    W[(p1 + k) * n + i][r1 + c] = W2[k * n + i][c]
        
        return U, V, W, r

    # Now build the 3x4x5 decomposition
    # Step 1: Split m=4 into 2+2
    # Sub-problem 1: 3x2x5
    # Sub-problem 2: 3x2x5
    
    # For each 3x2x5:
    # Split n=3 into 1+2
    # Sub-sub 1: 1x2x5 (schoolbook, rank 10)
    # Sub-sub 2: 2x2x5
    # For 2x2x5: split p=5 into 2+3
    # Sub-sub-sub 1: 2x2x2 (Strassen, rank 7)
    # Sub-sub-sub 2: 2x2x3 (schoolbook, rank 12)
    
    # Build 2x2x3 (schoolbook)
    U_223, V_223, W_223, r_223 = schoolbook(2, 2, 3)
    
    # Build 2x2x2 (Strassen)
    U_222, V_222, W_222, r_222 = strassen_222()
    
    # Combine 2x2x2 and 2x2x3 along p to get 2x2x5
    U_225, V_225, W_225, r_225 = combine_along_p(U_222, V_222, W_222, r_222, 2, 2, 2, 
                                                    U_223, V_223, W_223, r_223, 3)
    
    # Build 1x2x5 (schoolbook)
    U_125, V_125, W_125, r_125 = schoolbook(1, 2, 5)
    
    # Combine 1x2x5 and 2x2x5 along n to get 3x2x5
    U_325, V_325, W_325, r_325 = combine_along_n(U_125, V_125, W_125, r_125, 1, 
                                                   U_225, V_225, W_225, r_225, 2, 5)
    
    # Now we have two copies of 3x2x5 (for m1=2 and m2=2)
    # Combine them along m to get 3x4x5
    U_final, V_final, W_final, r_final = combine_along_m(U_325, V_325, W_325, r_325,
                                                          U_325, V_325, W_325, r_325,
                                                          3, 2, 2, 5)
    
    return {"U": U_final, "V": V_final, "W": W_final}
# EVOLVE-BLOCK-END
