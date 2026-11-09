"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 44 decomposition
    # Part 1: j=0, rank 16
    # Part 2: j=1,2, rank 28 (4 blocks of Strassen 2x2x2)
    
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A[:, 0] * B[0, :]
    # i in 0..1, k in 0..7
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    # Part 2: A[:, 1:3] * B[1:3, :]
    # Split p=8 into 4 blocks of 2
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # Strassen 7 multiplications
        # M1: (A01 + A12) * (B1_k0 + B2_k1)
        U[0 * m + 1][col] = 1
        U[1 * m + 2][col] = 1
        V[1 * p + k0][col] = 1
        V[2 * p + k1][col] = 1
        W[k0 * n + 0][col] = 1   # C0k0 += M1
        W[k1 * n + 1][col] = 1   # C1k1 += M1
        col += 1
        
        # M2: (A11 + A12) * B1_k0
        U[1 * m + 1][col] = 1
        U[1 * m + 2][col] = 1
        V[1 * p + k0][col] = 1
        W[k0 * n + 1][col] = 1   # C1k0 += M2
        col += 1
        
        # M3: A01 * (B1_k1 - B2_k1)
        U[0 * m + 1][col] = 1
        V[1 * p + k1][col] = 1
        V[2 * p + k1][col] = -1
        W[k1 * n + 0][col] = 1   # C0k1 += M3
        W[k1 * n + 1][col] = 1   # C1k1 += M3
        col += 1
        
        # M4: A12 * (B2_k0 - B1_k0)
        U[1 * m + 2][col] = 1
        V[2 * p + k0][col] = 1
        V[1 * p + k0][col] = -1
        W[k0 * n + 0][col] = 1   # C0k0 += M4
        W[k0 * n + 1][col] = 1   # C1k0 += M4
        col += 1
        
        # M5: (A01 + A02) * B2_k1
        U[0 * m + 1][col] = 1
        U[0 * m + 2][col] = 1
        V[2 * p + k1][col] = 1
        W[k0 * n + 0][col] = -1  # C0k0 -= M5
        W[k1 * n + 0][col] = 1   # C0k1 += M5
        col += 1
        
        # M6: (A11 - A01) * (B1_k0 + B1_k1)
        U[1 * m + 1][col] = 1
        U[0 * m + 1][col] = -1
        V[1 * p + k0][col] = 1
        V[1 * p + k1][col] = 1
        W[k1 * n + 1][col] = 1   # C1k1 += M6
        col += 1
        
        # M7: (A02 - A12) * (B2_k0 + B2_k1)
        U[0 * m + 2][col] = 1
        U[1 * m + 2][col] = -1
        V[2 * p + k0][col] = 1
        V[2 * p + k1][col] = 1
        W[k0 * n + 0][col] = 1   # C0k0 += M7
        col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
