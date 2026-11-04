DIMENSIONS = [2, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: j=0, schoolbook for 2x1 x 1x8 (16 terms)
    for i in range(n):
        for k in range(p):
            U[i * m][col] = 1
            V[k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    # Part 2: j=1,2, Strassen on 4 blocks of 2x2 x 2x2 (28 terms)
    for b in range(4):
        a11 = 1  # A[0,1]
        a12 = 2  # A[0,2]
        a21 = 4  # A[1,1]
        a22 = 5  # A[1,2]
        
        b11 = 8 + 2 * b
        b12 = 9 + 2 * b
        b21 = 16 + 2 * b
        b22 = 17 + 2 * b
        
        c11 = 4 * b
        c12 = 4 * b + 2
        c21 = 4 * b + 1
        c22 = 4 * b + 3
        
        # M1 = (A11+A22)(B11+B22)
        U[a11][col] = 1; U[a22][col] = 1
        V[b11][col] = 1; V[b22][col] = 1
        W[c11][col] = 1; W[c22][col] = 1
        col += 1
        
        # M2 = (A21+A22)B11
        U[a21][col] = 1; U[a22][col] = 1
        V[b11][col] = 1
        W[c21][col] = 1
        col += 1
        
        # M3 = A11(B12-B22)
        U[a11][col] = 1
        V[b12][col] = 1; V[b22][col] = -1
        W[c12][col] = 1
        col += 1
        
        # M4 = A22(B21-B11)
        U[a22][col] = 1
        V[b21][col] = 1; V[b11][col] = -1
        W[c11][col] = 1; W[c21][col] = 1
        col += 1
        
        # M5 = (A11+A12)B22
        U[a11][col] = 1; U[a12][col] = 1
        V[b22][col] = 1
        W[c11][col] = -1; W[c12][col] = 1
        col += 1
        
        # M6 = (A21-A11)(B11+B12)
        U[a21][col] = 1; U[a11][col] = -1
        V[b11][col] = 1; V[b12][col] = 1
        W[c22][col] = 1
        col += 1
        
        # M7 = (A12-A22)(B21+B22)
        U[a12][col] = 1; U[a22][col] = -1
        V[b21][col] = 1; V[b22][col] = 1
        W[c11][col] = 1; W[c22][col] = -1
        col += 1
    
    return {"U": U, "V": V, "W": W}
