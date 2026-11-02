"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We split the multiplication into 4 blocks:
    # A is 2x3, B is 3x10.
    # Split A into A1 (1x3) and A2 (1x3).
    # Split B into B1 (3x5) and B2 (3x5).
    # Products: A1B1 (1x5), A1B2 (1x5), A2B1 (1x5), A2B2 (1x5).
    # Each 1x3 * 3x5 product has rank 3.
    # Total rank 12.
    
    rank = 12
    U = [[0] * rank for _ in range(n * m)] # 6 x 12
    V = [[0] * rank for _ in range(m * p)] # 30 x 12
    W = [[0] * rank for _ in range(p * n)] # 20 x 12
    
    col = 0
    # Block 1: A1 (row 0) * B1 (cols 0-4)
    # A1 indices: 0 (i=0, j=0..2) -> U[0], U[1], U[2]
    # B1 indices: j=0..2, k=0..4 -> V[j*10 + k]
    # W indices: k=0..4, i=0 -> W[k*2 + 0]
    for j in range(3):
        U[j][col] = 1
        for k in range(5):
            V[j * p + k][col] = 1
            W[k * n][col] = 1
        col += 1
        
    # Block 2: A1 (row 0) * B2 (cols 5-9)
    for j in range(3):
        U[j][col] = 1
        for k in range(5, 10):
            V[j * p + k][col] = 1
            W[k * n][col] = 1
        col += 1
        
    # Block 3: A2 (row 1) * B1 (cols 0-4)
    # A2 indices: i=1, j=0..2 -> U[3], U[4], U[5]
    for j in range(3):
        U[3 + j][col] = 1
        for k in range(5):
            V[j * p + k][col] = 1
            W[k * n + 1][col] = 1
        col += 1
        
    # Block 4: A2 (row 1) * B2 (cols 5-9)
    for j in range(3):
        U[3 + j][col] = 1
        for k in range(5, 10):
            V[j * p + k][col] = 1
            W[k * n + 1][col] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
