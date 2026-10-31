"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    # Strassen factors for 2x2 matrix multiplication
    # Rows correspond to elements of A, B, C respectively.
    # Cols correspond to the 7 Strassen products.
    
    U2 = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    
    V2 = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    
    W2 = [
        [1, 0, 0, -1, 1, 0, -1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    
    # Construct 4x4 factors via Kronecker product
    # U is 16x49, V is 16x49, W is 16x49
    rank = 49
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]
    
    for i1 in range(2):
        for i2 in range(2):
            i = 2 * i1 + i2
            for r1 in range(7):
                for r2 in range(7):
                    r = 7 * r1 + r2
                    U[i][r] = U2[i1][r1] * U2[i2][r2]
                    V[i][r] = V2[i1][r1] * V2[i2][r2]
                    W[i][r] = W2[i1][r1] * W2[i2][r2]
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
