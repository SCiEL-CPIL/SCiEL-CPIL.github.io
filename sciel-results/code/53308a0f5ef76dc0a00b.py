import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    # Rank 7 decomposition for 2x2 matrix multiplication
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
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    
    n, m, p = DIMENSIONS
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(4):
        i1 = i // 2
        i2 = i % 2
        for j in range(4):
            j1 = j // 2
            j2 = j % 2
            for k in range(4):
                k1 = k // 2
                k2 = k % 2
                for c1 in range(7):
                    for c2 in range(7):
                        c = c1 * 7 + c2
                        U[i * 4 + j][c] = U2[i1 * 2 + j1][c1] * U2[i2 * 2 + j2][c2]
                        V[j * 4 + k][c] = V2[j1 * 2 + k1][c1] * V2[j2 * 2 + k2][c2]
                        W[k * 4 + i][c] = W2[k1 * 2 + i1][c1] * W2[k2 * 2 + i2][c2]
                        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
