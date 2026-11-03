"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 68 decomposition
    rank = 68
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen part for A[0:2, 0:2] * B[0:2, 0:8]
    # Split B columns into 4 pairs
    for c in range(4):
        k1 = 2 * c
        k2 = 2 * c + 1
        
        # Term 1
        U[0][col] = 1
        U[1][col] = 1
        V[k1][col] = 1
        V[8 + k2][col] = 1
        W[k1 * 3 + 0][col] = 1
        W[k2 * 3 + 1][col] = 1
        col += 1
        
        # Term 2
        U[3][col] = 1
        U[4][col] = 1
        V[k1][col] = 1
        W[k1 * 3 + 1][col] = 1
        col += 1
        
        # Term 3
        U[0][col] = 1
        V[8 + k1][col] = 1
        V[8 + k2][col] = -1
        W[k2 * 3 + 0][col] = 1
        col += 1
        
        # Term 4
        U[4][col] = 1
        V[k1][col] = 1
        V[k2][col] = -1
        W[k1 * 3 + 0][col] = 1
        W[k1 * 3 + 1][col] = 1
        col += 1
        
        # Term 5
        U[0][col] = 1
        U[1][col] = 1
        V[8 + k2][col] = 1
        W[k1 * 3 + 0][col] = -1
        W[k2 * 3 + 0][col] = 1
        col += 1
        
        # Term 6
        U[3][col] = 1
        U[0][col] = -1
        V[k1][col] = 1
        V[k2][col] = 1
        W[k2 * 3 + 1][col] = 1
        col += 1
        
        # Term 7
        U[1][col] = 1
        U[4][col] = -1
        V[8 + k1][col] = 1
        V[8 + k2][col] = 1
        W[k1 * 3 + 0][col] = 1
        W[k2 * 3 + 1][col] = 1
        col += 1

    # Schoolbook part: A[0:2, 2] * B[2, 0:8]
    # A[0,2] is U[2], A[1,2] is U[5]
    for i in range(2):
        for k in range(8):
            U[i * 3 + 2][col] = 1
            V[16 + k][col] = 1
            W[k * 3 + i][col] = 1
            col += 1

    # Schoolbook part: A[2, 0:2] * B[0:2, 0:8]
    # A[2,0] is U[6], A[2,1] is U[7]
    for j in range(2):
        for k in range(8):
            U[2 * 3 + j][col] = 1
            V[j * 8 + k][col] = 1
            W[k * 3 + 2][col] = 1
            col += 1

    # Schoolbook part: A[2,2] * B[2, 0:8]
    # A[2,2] is U[8]
    for k in range(8):
        U[8][col] = 1
        V[16 + k][col] = 1
        W[k * 3 + 2][col] = 1
        col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
