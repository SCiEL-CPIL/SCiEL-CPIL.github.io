"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 26
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    c = 0
    
    # Block 1a: A1 (rows 0,1) * B1a (cols 0,1)
    # i in {0,1}, j in {0,1,2}, k in {0,1}
    for i in [0, 1]:
        for j in range(m):
            for k in [0, 1]:
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
                
    # Block 1b: A1 (rows 0,1) * B1b (cols 2,3)
    # i in {0,1}, j in {0,1,2}, k in {2,3}
    for i in [0, 1]:
        for j in range(m):
            for k in [2, 3]:
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
                
    # Block 2a: A1 (rows 0,1) * B2a (cols 4,5)
    # i in {0,1}, j in {0,1,2}, k in {4,5}
    for i in [0, 1]:
        for j in range(m):
            for k in [4, 5]:
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
                
    # Block 2b: A1 (rows 0,1) * B2b (col 6)
    # i in {0,1}, j in {0,1,2}, k = 6
    # Rank 1 term
    for i in [0, 1]:
        for j in range(m):
            U[i * m + j][c] = 1
            V[j * p + 6][c] = 1
            W[6 * n + i][c] = 1
    c += 1
    
    # Block 3: A2 (row 2) * B1 (cols 0,1,2,3)
    # i = 2, j in {0,1,2}, k in {0,1,2,3}
    # 4 terms, one for each k
    for k in [0, 1, 2, 3]:
        for j in range(m):
            U[2 * m + j][c] = 1
            V[j * p + k][c] = 1
            W[k * n + 2][c] = 1
        c += 1
        
    # Block 4: A2 (row 2) * B2 (cols 4,5,6)
    # i = 2, j in {0,1,2}, k in {4,5,6}
    # 3 terms, one for each k
    for k in [4, 5, 6]:
        for j in range(m):
            U[2 * m + j][c] = 1
            V[j * p + k][c] = 1
            W[k * n + 2][c] = 1
        c += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
