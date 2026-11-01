"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=10
    num_groups = p // 2  # 5 groups of 2 columns each
    rank = num_groups * 7  # 35
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x 35
    V = [[0] * rank for _ in range(m * p)]  # 20 x 35
    W = [[0] * rank for _ in range(p * n)]  # 20 x 35
    
    for g in range(num_groups):
        k0 = 2 * g      # first column in group
        k1 = 2 * g + 1  # second column in group
        base = g * 7    # starting column index in U, V, W
        
        # P1 = (A[0,0] + A[1,1]) * (B[0,k0] + B[1,k1])
        c = base + 0
        U[0][c] = 1
        U[3][c] = 1
        V[k0][c] = 1
        V[10 + k1][c] = 1
        W[4 * g][c] = 1       # C[0,k0]
        W[4 * g + 3][c] = 1   # C[1,k1]
        
        # P2 = (A[1,0] + A[1,1]) * B[0,k0]
        c = base + 1
        U[2][c] = 1
        U[3][c] = 1
        V[k0][c] = 1
        W[4 * g + 1][c] = 1   # C[1,k0]
        W[4 * g + 3][c] = -1  # C[1,k1]
        
        # P3 = A[0,0] * (B[0,k1] - B[1,k1])
        c = base + 2
        U[0][c] = 1
        V[k1][c] = 1
        V[10 + k1][c] = -1
        W[4 * g + 2][c] = 1   # C[0,k1]
        W[4 * g + 3][c] = 1   # C[1,k1]
        
        # P4 = A[1,1] * (B[1,k0] - B[0,k0])
        c = base + 3
        U[3][c] = 1
        V[10 + k0][c] = 1
        V[k0][c] = -1
        W[4 * g][c] = 1       # C[0,k0]
        W[4 * g + 1][c] = 1   # C[1,k0]
        
        # P5 = (A[0,0] + A[0,1]) * B[1,k1]
        c = base + 4
        U[0][c] = 1
        U[1][c] = 1
        V[10 + k1][c] = 1
        W[4 * g][c] = -1      # C[0,k0]
        W[4 * g + 2][c] = 1   # C[0,k1]
        
        # P6 = (A[1,0] - A[0,0]) * (B[0,k0] + B[0,k1])
        c = base + 5
        U[2][c] = 1
        U[0][c] = -1
        V[k0][c] = 1
        V[k1][c] = 1
        W[4 * g + 3][c] = 1   # C[1,k1]
        
        # P7 = (A[0,1] - A[1,1]) * (B[1,k0] + B[1,k1])
        c = base + 6
        U[1][c] = 1
        U[3][c] = -1
        V[10 + k0][c] = 1
        V[10 + k1][c] = 1
        W[4 * g][c] = 1       # C[0,k0]
        W[4 * g + 3][c] = 1   # C[1,k1]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
