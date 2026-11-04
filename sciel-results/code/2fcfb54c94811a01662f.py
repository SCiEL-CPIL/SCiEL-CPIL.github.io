"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=10
    
    rank = 50
    U = [[0] * rank for _ in range(n * m)]   # 6 x rank
    V = [[0] * rank for _ in range(m * p)]   # 30 x rank
    W = [[0] * rank for _ in range(p * n)]   # 20 x rank
    
    col = 0
    
    # Part 1: Row 2 of A contributions (A02, A12 with B2,0..9)
    # 20 terms: each output C_i,j gets one term from A_i2 * B_2,j
    for j in range(10):
        # C0,j += A02 * B2,j
        U[2][col] = 1
        V[20 + j][col] = 1
        W[2*j][col] = 1
        col += 1
        # C1,j += A12 * B2,j
        U[5][col] = 1
        V[20 + j][col] = 1
        W[2*j+1][col] = 1
        col += 1
    # col = 20
    
    # Part 2: Core 2x2x10 using 30 terms
    # Use a rank-6 decomposition per column-pair, 5 pairs = 30 terms
    for t in range(5):
        j0 = 2 * t
        j1 = 2 * t + 1
        
        # T1: (A00+A11) * (B0,j0 + B1,j1) -> C0,j0 + C1,j1
        U[0][col] += 1
        U[3][col] += 1
        V[j0][col] += 1
        V[10+j1][col] += 1
        W[2*j0][col] += 1
        W[2*j1+1][col] += 1
        col += 1
        
        # T2: (A10+A11) * B0,j0 -> C1,j0 - C1,j1
        U[3][col] += 1
        U[4][col] += 1
        V[j0][col] += 1
        W[2*j0+1][col] += 1
        W[2*j1+1][col] -= 1
        col += 1
        
        # T3: A00 * (B0,j1 - B1,j1) -> C0,j1 + C1,j1
        U[0][col] += 1
        V[10+j1][col] += 1
        V[10+j1][col] -= 1
        W[2*j1][col] += 1
        W[2*j1+1][col] += 1
        col += 1
        
        # T4: A11 * (B1,j0 - B0,j0) -> C0,j0 + C1,j0
        U[3][col] += 1
        V[10+j0][col] += 1
        V[j0][col] -= 1
        W[2*j0][col] += 1
        W[2*j0+1][col] += 1
        col += 1
        
        # T5: (A00+A01) * B1,j1 -> -C0,j0 + C0,j1
        U[0][col] += 1
        U[1][col] += 1
        V[10+j1][col] += 1
        W[2*j0][col] -= 1
        W[2*j1][col] += 1
        col += 1
        
        # T6: (A10-A00) * (B0,j0 + B0,j1) -> C1,j1
        U[3][col] += 1
        U[0][col] -= 1
        V[j0][col] += 1
        V[j1][col] += 1
        W[2*j1+1][col] += 1
        col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
