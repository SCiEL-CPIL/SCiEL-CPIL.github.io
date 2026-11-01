DIMENSIONS = [2, 2, 9]

def construct():
    n, m, p = DIMENSIONS  # 2, 2, 9
    rank = 29
    U = [[0] * rank for _ in range(n * m)]  # 4 x 29
    V = [[0] * rank for _ in range(m * p)]  # 18 x 29
    W = [[0] * rank for _ in range(p * n)]  # 18 x 29
    
    col = 0
    
    # Strassen blocks for column pairs
    for block in range(4):
        k0 = 2 * block
        k1 = 2 * block + 1
        
        # M1 = (A00+A11)(B[0,k0]+B[1,k1])
        U[0][col] = 1; U[3][col] = 1
        V[k0][col] = 1; V[m + k1][col] = 1
        W[k0*n + 0][col] = 1; W[k1*n + 1][col] = 1
        col += 1
        
        # M2 = (A10+A11)B[0,k0]
        U[2][col] = 1; U[3][col] = 1
        V[k0][col] = 1
        W[k0*n + 1][col] = 1; W[k1*n + 1][col] = -1
        col += 1
        
        # M3 = A00(B[1,k0]-B[1,k1])
        U[0][col] = 1
        V[m + k0][col] = 1; V[m + k1][col] = -1
        W[k1*n + 0][col] = 1; W[k1*n + 1][col] = 1
        col += 1
        
        # M4 = A11(B[0,k0]-B[0,k1])
        U[3][col] = 1
        V[k0][col] = 1; V[k1][col] = -1
        W[k0*n + 0][col] = 1; W[k0*n + 1][col] = 1
        col += 1
        
        # M5 = (A00+A01)B[1,k1]
        U[0][col] = 1; U[1][col] = 1
        V[m + k1][col] = 1
        W[k0*n + 0][col] = -1; W[k1*n + 0][col] = 1
        col += 1
        
        # M6 = (A10-A00)(B[0,k0]+B[0,k1])
        U[2][col] = 1; U[0][col] = -1
        V[k0][col] = 1; V[k1][col] = 1
        W[k1*n + 1][col] = 1
        col += 1
        
        # M7 = (A01-A11)(B[0,k1]+B[1,k1])
        U[1][col] = 1; U[3][col] = -1
        V[k1][col] = 1; V[m + k1][col] = 1
        W[k0*n + 0][col] = 1
        col += 1
    
    # Rank-1 term for last column (k=8): T[i,j,8] = 1 for all i,j
    k = 8
    U[0][col] = 1; U[1][col] = 1; U[2][col] = 1; U[3][col] = 1
    V[k][col] = 1; V[m + k][col] = 1
    W[k*n + 0][col] = 1; W[k*n + 1][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
