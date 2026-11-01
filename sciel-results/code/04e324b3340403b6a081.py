"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 21 decomposition
    rank = 21
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Sub-block 1: i in {0,1}, j in {0,1}, k in {0,1}
    for i in range(2):
        for j in range(2):
            for k in range(2):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    # Sub-block 2: i in {0,1}, j in {0,1}, k in {2,3}
    for i in range(2):
        for j in range(2):
            for k in range(2, 4):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    # Sub-block 3: i in {0,1}, j in {0,1}, k in {4}
    for i in range(2):
        for j in range(2):
            k = 4
            U[i * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    # Sub-block 4: i in {0,1}, j in {2}, k in {0..4}
    # Rank 1 term
    U[0 * m + 2][col] = 1
    U[1 * m + 2][col] = 1
    for k in range(5):
        V[2 * p + k][col] = 1
        W[k * n + 0][col] = 1
        W[k * n + 1][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
