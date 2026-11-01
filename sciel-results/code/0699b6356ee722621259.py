DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 21
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen U vectors (shape 4)
    # Indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    U_strassen = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    
    for r in range(3):
        base_col = 2 * r
        for s in range(7):
            col = r * 7 + s
            for i in range(4):
                U[i][col] = U_strassen[s][i]
            
            if s == 0:
                V[base_col][col] = 1
                V[p + base_col + 1][col] = 1
                W[base_col * n][col] = 1
                W[(base_col + 1) * n + 1][col] = 1
            elif s == 1:
                V[base_col][col] = 1
                W[base_col * n + 1][col] = 1
                W[(base_col + 1) * n + 1][col] = -1
            elif s == 2:
                V[base_col + 1][col] = 1
                V[p + base_col + 1][col] = -1
                W[(base_col + 1) * n][col] = 1
                W[(base_col + 1) * n + 1][col] = 1
            elif s == 3:
                V[p + base_col][col] = 1
                V[base_col][col] = -1
                W[base_col * n][col] = 1
                W[base_col * n + 1][col] = 1
            elif s == 4:
                V[p + base_col + 1][col] = 1
                W[base_col * n][col] = -1
                W[(base_col + 1) * n][col] = 1
            elif s == 5:
                V[base_col][col] = 1
                V[base_col + 1][col] = 1
                W[(base_col + 1) * n + 1][col] = 1
            elif s == 6:
                V[p + base_col][col] = 1
                V[p + base_col + 1][col] = 1
                W[base_col * n][col] = 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
