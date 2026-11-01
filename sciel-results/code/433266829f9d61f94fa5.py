"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank calculation: 5 blocks of 2x2 (rank 7 each) + 1 block of 2x1 (rank 4)
    # 5 * 7 + 4 = 39
    rank = 39
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen_block(t):
        # Block t corresponds to columns 2*t and 2*t+1
        k0 = 2 * t
        k1 = 2 * t + 1
        base_col = 0
        
        # Term 1: (A00+A11)(B0k0+B1k1)
        col = base_col
        U[0][col] += 1; U[3][col] += 1
        V[0*p+k0][col] += 1; V[1*p+k1][col] += 1
        W[k0*n+0][col] += 1; W[k1*n+1][col] += 1
        
        # Term 2: (A10+A11)B0k0
        col = base_col + 1
        U[2][col] += 1; U[3][col] += 1
        V[0*p+k0][col] += 1
        W[k0*n+1][col] += 1; W[k1*n+1][col] -= 1
        
        # Term 3: A00(B1k0-B1k1)
        col = base_col + 2
        U[0][col] += 1
        V[1*p+k0][col] += 1; V[1*p+k1][col] -= 1
        W[k1*n+0][col] += 1; W[k1*n+1][col] += 1
        
        # Term 4: A11(B0k1-B0k0)
        col = base_col + 3
        U[3][col] += 1
        V[0*p+k1][col] += 1; V[0*p+k0][col] -= 1
        W[k0*n+0][col] += 1; W[k0*n+1][col] += 1
        
        # Term 5: (A00+A01)B1k1
        col = base_col + 4
        U[0][col] += 1; U[1][col] += 1
        V[1*p+k1][col] += 1
        W[k0*n+0][col] -= 1; W[k1*n+0][col] += 1
        
        # Term 6: (A10-A00)(B0k0+B0k1)
        col = base_col + 5
        U[2][col] += 1; U[0][col] -= 1
        V[0*p+k0][col] += 1; V[0*p+k1][col] += 1
        W[k1*n+1][col] += 1
        
        # Term 7: (A01-A11)(B1k0+B1k1)
        col = base_col + 6
        U[1][col] += 1; U[3][col] -= 1
        V[1*p+k0][col] += 1; V[1*p+k1][col] += 1
        W[k0*n+0][col] += 1

    # Add 5 Strassen blocks
    for t in range(5):
        add_strassen_block(t)
        
    # Add last column (k=10) with rank 4
    # C0,10 = A00*B0,10 + A01*B1,10
    # C1,10 = A10*B0,10 + A11*B1,10
    base_col = 35
    k_last = 10
    
    # Term 1: A00 * B0,10
    col = base_col
    U[0][col] = 1
    V[0*p+k_last][col] = 1
    W[k_last*n+0][col] = 1
    
    # Term 2: A01 * B1,10
    col = base_col + 1
    U[1][col] = 1
    V[1*p+k_last][col] = 1
    W[k_last*n+0][col] = 1
    
    # Term 3: A10 * B0,10
    col = base_col + 2
    U[2][col] = 1
    V[0*p+k_last][col] = 1
    W[k_last*n+1][col] = 1
    
    # Term 4: A11 * B1,10
    col = base_col + 3
    U[3][col] = 1
    V[1*p+k_last][col] = 1
    W[k_last*n+1][col] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
