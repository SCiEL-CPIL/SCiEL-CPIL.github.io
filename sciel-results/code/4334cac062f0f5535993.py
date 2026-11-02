"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 68 decomposition using Strassen for 2x2 blocks and schoolbook for the rest
    rank = 68
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen definitions
    # M_s: (L_A, L_B, W_coeffs)
    # L_A: list of (i, j, coeff)
    # L_B: list of (j, k_local, coeff)
    # W_coeffs: list of (i, k_local, coeff)
    
    strassen_blocks = [
        # M1
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        # M2
        ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)]),
        # M3
        ([(0,0,1)], [(1,0,1), (1,1,-1)], [(0,1,1), (1,1,1)]),
        # M4
        ([(1,1,1)], [(0,1,1), (0,0,-1)], [(0,0,1), (1,0,1)]),
        # M5
        ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)]),
        # M6
        ([(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)], [(1,1,1)]),
        # M7
        ([(0,1,1), (1,1,-1)], [(1,1,1), (0,1,1)], [(0,0,1)])
    ]
    
    # 4 Strassen blocks for A[0:2, 0:2] * B[0:2, k:k+2]
    offsets = [0, 2, 4, 6]
    for k_off in offsets:
        for la, lb, wc in strassen_blocks:
            # Add U
            for i, j, coeff in la:
                U[i * m + j][col] += coeff
            # Add V
            for j, k_loc, coeff in lb:
                V[j * p + (k_off + k_loc)][col] += coeff
            # Add W
            for i, k_loc, coeff in wc:
                W[(k_off + k_loc) * n + i][col] += coeff
            col += 1
            
    # Schoolbook terms for the remaining blocks
    # A[0:2, 2] * B[2, 0:4] -> i in 0,1; j=2; k in 0..3
    for i in range(2):
        for k in range(4):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    # A[0:2, 2] * B[2, 4:8] -> i in 0,1; j=2; k in 4..7
    for i in range(2):
        for k in range(4, 8):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    # A[2, 0:2] * B[0:2, 0:4] -> i=2; j in 0,1; k in 0..3
    for j in range(2):
        for k in range(4):
            U[2 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 2][col] = 1
            col += 1
            
    # A[2, 0:2] * B[0:2, 4:8] -> i=2; j in 0,1; k in 4..7
    for j in range(2):
        for k in range(4, 8):
            U[2 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 2][col] = 1
            col += 1
            
    # A[2, 2] * B[2, 0:4] -> i=2; j=2; k in 0..3
    for k in range(4):
        U[2 * m + 2][col] = 1
        V[2 * p + k][col] = 1
        W[k * n + 2][col] = 1
        col += 1
        
    # A[2, 2] * B[2, 4:8] -> i=2; j=2; k in 4..7
    for k in range(4, 8):
        U[2 * m + 2][col] = 1
        V[2 * p + k][col] = 1
        W[k * n + 2][col] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
