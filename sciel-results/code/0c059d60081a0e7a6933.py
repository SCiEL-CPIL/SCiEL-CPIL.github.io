DIMENSIONS = [2, 3, 8]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=8
    rank = 32
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # Strassen M1 = (A00+A11)(B0k0+B1k1)
        U[0*m+0][col] = 1; U[1*m+1][col] = 1
        V[0*p+k0][col] = 1; V[1*p+k1][col] = 1
        W[k0*n+0][col] = 1; W[k1*n+1][col] = 1
        col += 1
        
        # Strassen M2 = (A10+A11)(B0k0)
        U[1*m+0][col] = 1; U[1*m+1][col] = 1
        V[0*p+k0][col] = 1
        W[k0*n+1][col] = 1; W[k1*n+1][col] = -1
        col += 1
        
        # Strassen M3 = A00(B0k1-B1k1)
        U[0*m+0][col] = 1
        V[0*p+k1][col] = 1; V[1*p+k1][col] = -1
        W[k1*n+0][col] = 1; W[k1*n+1][col] = 1
        col += 1
        
        # Strassen M4 = A11(B1k0-B0k0)
        U[1*m+1][col] = 1
        V[1*p+k0][col] = 1; V[0*p+k0][col] = -1
        W[k0*n+0][col] = 1; W[k0*n+1][col] = 1
        col += 1
        
        # Strassen M5 = (A00+A01)(B1k1)
        U[0*m+0][col] = 1; U[0*m+1][col] = 1
        V[1*p+k1][col] = 1
        W[k0*n+0][col] = -1; W[k1*n+0][col] = 1
        col += 1
        
        # Strassen M6 = (A10-A00)(B0k0+B0k1)
        U[1*m+0][col] = 1; U[0*m+0][col] = -1
        V[0*p+k0][col] = 1; V[0*p+k1][col] = 1
        W[k1*n+1][col] = 1
        col += 1
        
        # Strassen M7 = (A01-A11)(B1k0+B1k1)
        U[0*m+1][col] = 1; U[1*m+1][col] = -1
        V[1*p+k0][col] = 1; V[1*p+k1][col] = 1
        W[k0*n+0][col] = 1
        col += 1
        
        # Rank-1: 2x1 * 1x2
        U[0*m+2][col] = 1; U[1*m+2][col] = 1
        V[2*p+k0][col] = 1; V[2*p+k1][col] = 1
        W[k0*n+0][col] = 1; W[k0*n+1][col] = 1
        W[k1*n+0][col] = 1; W[k1*n+1][col] = 1
        col += 1
    
    return {"U": U, "V": V, "W": W}
