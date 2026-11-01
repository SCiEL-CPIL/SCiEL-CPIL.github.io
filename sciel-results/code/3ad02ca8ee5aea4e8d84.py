"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
def construct():
    # Rank-17 decomposition using Strassen-style blocks for k in {0,1} and {2,3},
    # plus schoolbook for k=4.
    n, m, p = DIMENSIONS  # n=2, m=2, p=5
    rank = 17
    U = [[0] * rank for _ in range(n * m)]   # 4 x 17
    V = [[0] * rank for _ in range(m * p)]   # 10 x 17
    W = [[0] * rank for _ in range(p * n)]   # 10 x 17

    def strassen_block(k0, k1, col0):
        """Add 7 Strassen-style terms for columns k0 and k1."""
        # A indices: (i,j) -> i*m+j
        # B indices: (j,k) -> j*p+k
        # C indices: (k,i) -> k*n+i  (trace convention W[k*n+i, r])
        
        # M1 = (A00+A11)(B0k0+B1k1)
        U[0][col0] = 1; U[3][col0] = 1
        V[0*p+k0][col0] = 1; V[1*p+k1][col0] = 1
        W[k0*n+0][col0] = 1; W[k1*n+1][col0] = 1
        col0 += 1
        
        # M2 = (A10+A11) B0k0
        U[2][col0] = 1; U[3][col0] = 1
        V[0*p+k0][col0] = 1
        W[k0*n+1][col0] = 1
        col0 += 1
        
        # M3 = A00 (B0k1 - B1k1)
        U[0][col0] = 1
        V[0*p+k1][col0] = 1; V[1*p+k1][col0] = -1
        W[k1*n+0][col0] = 1
        col0 += 1
        
        # M4 = A11 (B1k0 - B0k0)
        U[3][col0] = 1
        V[1*p+k0][col0] = 1; V[0*p+k0][col0] = -1
        W[k0*n+1][col0] = 1
        col0 += 1
        
        # M5 = (A00+A01) B1k1
        U[0][col0] = 1; U[1][col0] = 1
        V[1*p+k1][col0] = 1
        W[k1*n+0][col0] = -1  # contributes -M5 to C[0,k1]
        col0 += 1
        
        # M6 = (A10-A00)(B0k0+B0k1)
        U[2][col0] = 1; U[0][col0] = -1
        V[0*p+k0][col0] = 1; V[0*p+k1][col0] = 1
        W[k1*n+1][col0] = 1
        col0 += 1
        
        # M7 = (A01-A11)(B1k0+B1k1)
        U[1][col0] = 1; U[3][col0] = -1
        V[1*p+k0][col0] = 1; V[1*p+k1][col0] = 1
        W[k0*n+0][col0] = 1; W[k1*n+1][col0] = 1
        col0 += 1
        
        return col0

    col = 0
    # Block 1: k0=0, k1=1
    col = strassen_block(0, 1, col)
    # Block 2: k0=2, k1=3
    col = strassen_block(2, 3, col)
    
    # Schoolbook for k=4: 4 terms
    for i in range(n):
        for j in range(m):
            U[i*m+j][col] = 1
            V[j*p+4][col] = 1
            W[4*n+i][col] = 1
            col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
