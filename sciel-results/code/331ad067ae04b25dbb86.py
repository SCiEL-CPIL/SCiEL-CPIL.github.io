"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 8
    # Four 2x3x2 blocks, each with rank 11
    rank = 44
    U = [[0] * rank for _ in range(n * m)]  # 6 x 44
    V = [[0] * rank for _ in range(m * p)]  # 24 x 44
    W = [[0] * rank for _ in range(p * n)]  # 16 x 44
    
    # For each block t (k = 2t, 2t+1), define 11 rank-1 terms
    # Term structure for 2x3x2:
    # 0: (a00, b00) -> C[0,0]
    # 1: (a01, b10) -> C[0,0]
    # 2: (a02, b20) -> C[0,0]
    # 3: (a10, b00) -> C[1,0]
    # 4: (a11, b10) -> C[1,0]
    # 5: (a12, b20) -> C[1,0]
    # 6: (a00+a10, b01) -> C[0,1] + C[1,1]
    # 7: (a01, b11) -> C[0,1]
    # 8: (a02, b21) -> C[0,1]
    # 9: (a11, b11) -> C[1,1]
    # 10: (a12, b21) -> C[1,1]
    
    for t in range(4):
        base = t * 11
        k0 = 2 * t      # first k index in this block
        k1 = 2 * t + 1  # second k index in this block
        
        # Term 0: a[0,0]*b[0,k0] -> C[0,k0]
        U[0][base + 0] = 1       # a00
        V[0 * p + k0][base + 0] = 1  # b[0,k0]
        W[k0 * n + 0][base + 0] = 1  # C[0,k0]
        
        # Term 1: a[0,1]*b[1,k0] -> C[0,k0]
        U[1][base + 1] = 1       # a01
        V[1 * p + k0][base + 1] = 1  # b[1,k0]
        W[k0 * n + 0][base + 1] = 1  # C[0,k0]
        
        # Term 2: a[0,2]*b[2,k0] -> C[0,k0]
        U[2][base + 2] = 1       # a02
        V[2 * p + k0][base + 2] = 1  # b[2,k0]
        W[k0 * n + 0][base + 2] = 1  # C[0,k0]
        
        # Term 3: a[1,0]*b[0,k0] -> C[1,k0]
        U[3][base + 3] = 1       # a10
        V[0 * p + k0][base + 3] = 1  # b[0,k0]
        W[k0 * n + 1][base + 3] = 1  # C[1,k0]
        
        # Term 4: a[1,1]*b[1,k0] -> C[1,k0]
        U[4][base + 4] = 1       # a11
        V[1 * p + k0][base + 4] = 1  # b[1,k0]
        W[k0 * n + 1][base + 4] = 1  # C[1,k0]
        
        # Term 5: a[1,2]*b[2,k0] -> C[1,k0]
        U[5][base + 5] = 1       # a12
        V[2 * p + k0][base + 5] = 1  # b[2,k0]
        W[k0 * n + 1][base + 5] = 1  # C[1,k0]
        
        # Term 6: (a00+a10)*b[0,k1] -> C[0,k1] + C[1,k1]
        U[0][base + 6] = 1       # a00
        U[3][base + 6] = 1       # a10
        V[0 * p + k1][base + 6] = 1  # b[0,k1]
        W[k1 * n + 0][base + 6] = 1  # C[0,k1]
        W[k1 * n + 1][base + 6] = 1  # C[1,k1]
        
        # Term 7: a[0,1]*b[1,k1] -> C[0,k1]
        U[1][base + 7] = 1       # a01
        V[1 * p + k1][base + 7] = 1  # b[1,k1]
        W[k1 * n + 0][base + 7] = 1  # C[0,k1]
        
        # Term 8: a[0,2]*b[2,k1] -> C[0,k1]
        U[2][base + 8] = 1       # a02
        V[2 * p + k1][base + 8] = 1  # b[2,k1]
        W[k1 * n + 0][base + 8] = 1  # C[0,k1]
        
        # Term 9: a[1,1]*b[1,k1] -> C[1,k1]
        U[4][base + 9] = 1       # a11
        V[1 * p + k1][base + 9] = 1  # b[1,k1]
        W[k1 * n + 1][base + 9] = 1  # C[1,k1]
        
        # Term 10: a[1,2]*b[2,k1] -> C[1,k1]
        U[5][base + 10] = 1      # a12
        V[2 * p + k1][base + 10] = 1  # b[2,k1]
        W[k1 * n + 1][base + 10] = 1  # C[1,k1]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
