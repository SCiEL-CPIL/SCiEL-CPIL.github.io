# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Decompose p=9 into 4, 4, 1
    # Rank = 4*4*4 + 4*4*4 + 4*4*1 = 16 + 16 + 4 = 36
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Block 1: k=0..3
    for i in range(n):
        for j in range(m):
            for k in range(4):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    # Block 2: k=4..7
    for i in range(n):
        for j in range(m):
            for k in range(4):
                U[i * m + j][col] = 1
                V[j * p + 4 + k][col] = 1
                W[(4 + k) * n + i][col] = 1
                col += 1
    # Block 3: k=8
    for i in range(n):
        for j in range(m):
            U[i * m + j][col] = 1
            V[j * p + 8][col] = 1
            W[8 * n + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
