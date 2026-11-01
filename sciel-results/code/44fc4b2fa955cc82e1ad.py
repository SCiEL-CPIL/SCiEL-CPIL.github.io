"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 6
    rank = 21  # 3 Strassen blocks of 7
    
    U = [[0.0] * rank for _ in range(n * m)]  # (4, 21)
    V = [[0.0] * rank for _ in range(m * p)]  # (12, 21)
    W = [[0.0] * rank for _ in range(p * n)]  # (12, 21)
    
    col = 0
    for b in range(3):  # 3 blocks of 2 columns
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # M1 = (A00+A11)(B0k0+B1k1)
        U[0][col] += 1  # A[0,0]
        U[3][col] += 1  # A[1,1]
        V[0*p+k0][col] += 1  # B[0,k0]
        V[1*p+k1][col] += 1  # B[1,k1]
        W[k0*n+0][col] += 1  # C[0,k0]
        W[k1*n+1][col] += 1  # C[1,k1]
        col += 1
        
        # M2 = (A10+A11)B0k0
        U[2][col] += 1  # A[1,0]
        U[3][col] += 1  # A[1,1]
        V[0*p+k0][col] += 1  # B[0,k0]
        W[k0*n+1][col] += 1  # C[1,k0]
        W[k1*n+1][col] += -1  # C[1,k1]
        col += 1
        
        # M3 = A00(B0k1-B1k1)
        U[0][col] += 1  # A[0,0]
        V[0*p+k1][col] += 1  # B[0,k1]
        V[1*p+k1][col] += -1  # B[1,k1]
        W[k1*n+0][col] += 1  # C[0,k1]
        W[k1*n+1][col] += 1  # C[1,k1]
        col += 1
        
        # M4 = A11(B1k0-B0k0)
        U[3][col] += 1  # A[1,1]
        V[1*p+k0][col] += 1  # B[1,k0]
        V[0*p+k0][col] += -1  # B[0,k0]
        W[k0*n+0][col] += 1  # C[0,k0]
        W[k0*n+1][col] += 1  # C[1,k0]
        col += 1
        
        # M5 = (A00+A01)B1k1
        U[0][col] += 1  # A[0,0]
        U[1][col] += 1  # A[0,1]
        V[1*p+k1][col] += 1  # B[1,k1]
        W[k0*n+0][col] += -1  # C[0,k0]
        W[k1*n+0][col] += 1  # C[0,k1]
        col += 1
        
        # M6 = (A10-A00)(B0k0+B1k0)
        U[2][col] += 1  # A[1,0]
        U[0][col] += -1  # A[0,0]
        V[0*p+k0][col] += 1  # B[0,k0]
        V[1*p+k0][col] += 1  # B[1,k0]
        W[k1*n+1][col] += 1  # C[1,k1]
        col += 1
        
        # M7 = (A01-A11)(B0k1+B1k1)
        U[1][col] += 1  # A[0,1]
        U[3][col] += -1  # A[1,1]
        V[0*p+k1][col] += 1  # B[0,k1]
        V[1*p+k1][col] += 1  # B[1,k1]
        W[k0*n+0][col] += 1  # C[0,k0]
        W[k1*n+1][col] += -1  # C[1,k1]
        col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
