"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    """
    Tensor decomposition for 2x2 by 2x12 matrix multiplication.
    Uses Strassen's algorithm applied to 6 independent 2x2 blocks of the B matrix.
    Each 2x2 * 2x2 multiplication uses 7 rank-1 terms (Strassen).
    Total rank = 6 * 7 = 42.
    
    U shape: (n*m, rank) = (4, 42)
    V shape: (m*p, rank) = (24, 42)
    W shape: (p*n, rank) = (24, 42)
    
    Indexing conventions:
    U index: i*m + j  (i in [0,2), j in [0,2))
    V index: j*p + k  (j in [0,2), k in [0,12))
    W index: k*n + i  (k in [0,12), i in [0,2))
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=12
    
    # Strassen's 7 multiplications for 2x2 * 2x2:
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01], [b10, b11]]
    # C = A * B
    #
    # M1 = (a00 + a11) * (b00 + b11)
    # M2 = (a10 + a11) * b00
    # M3 = a00 * (b01 - b11)
    # M4 = a11 * (b10 - b00)
    # M5 = (a00 + a01) * b11
    # M6 = (a10 - a00) * (b00 + b01)
    # M7 = (a01 - a11) * (b10 + b11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]  # 4 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 24 x 42
    
    # For each block g (0 to 5), B columns are 2g and 2g+1
    # B[0, 2g] -> V index 0*p + 2g = 2g
    # B[0, 2g+1] -> V index 0*p + (2g+1) = 2g+1
    # B[1, 2g] -> V index 1*p + 2g = 12 + 2g
    # B[1, 2g+1] -> V index 1*p + (2g+1) = 12 + 2g + 1
    
    # C[0, 2g] -> W index 2g*n + 0 = 4g
    # C[0, 2g+1] -> W index (2g+1)*n + 0 = 4g + 2
    # C[1, 2g] -> W index 2g*n + 1 = 4g + 1
    # C[1, 2g+1] -> W index (2g+1)*n + 1 = 4g + 3
    
    # A indices in U:
    # a00 -> U[0]
    # a01 -> U[1]
    # a10 -> U[2]
    # a11 -> U[3]
    
    col = 0
    for g in range(p // 2):  # 6 blocks
        # V indices for this block
        v_b00 = 2 * g          # B[0, 2g]
        v_b01 = 2 * g + 1      # B[0, 2g+1]
        v_b10 = 12 + 2 * g     # B[1, 2g]
        v_b11 = 12 + 2 * g + 1 # B[1, 2g+1]
        
        # W indices for this block
        w_c00 = 4 * g          # C[0, 2g]
        w_c01 = 4 * g + 2      # C[0, 2g+1]
        w_c10 = 4 * g + 1      # C[1, 2g]
        w_c11 = 4 * g + 3      # C[1, 2g+1]
        
        # M1 = (a00 + a11) * (b00 + b11)
        U[0][col] = 1
        U[3][col] = 1
        V[v_b00][col] = 1
        V[v_b11][col] = 1
        W[w_c00][col] = 1
        W[w_c11][col] = 1
        col += 1
        
        # M2 = (a10 + a11) * b00
        U[2][col] = 1
        U[3][col] = 1
        V[v_b00][col] = 1
        W[w_c10][col] = 1
        W[w_c11][col] = -1
        col += 1
        
        # M3 = a00 * (b01 - b11)
        U[0][col] = 1
        V[v_b01][col] = 1
        V[v_b11][col] = -1
        W[w_c01][col] = 1
        W[w_c11][col] = 1
        col += 1
        
        # M4 = a11 * (b10 - b00)
        U[3][col] = 1
        V[v_b10][col] = 1
        V[v_b00][col] = -1
        W[w_c00][col] = 1
        W[w_c10][col] = 1
        col += 1
        
        # M5 = (a00 + a01) * b11
        U[0][col] = 1
        U[1][col] = 1
        V[v_b11][col] = 1
        W[w_c00][col] = -1
        W[w_c01][col] = 1
        col += 1
        
        # M6 = (a10 - a00) * (b00 + b01)
        U[2][col] = 1
        U[0][col] = -1
        V[v_b00][col] = 1
        V[v_b01][col] = 1
        W[w_c11][col] = 1
        col += 1
        
        # M7 = (a01 - a11) * (b10 + b11)
        U[1][col] = 1
        U[3][col] = -1
        V[v_b10][col] = 1
        V[v_b11][col] = 1
        W[w_c00][col] = 1
        col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
