import numpy as np

DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p  # 32
    
    # Initialize matrices with zeros
    U = np.zeros((n * m, rank), dtype=int)  # 4 x 32
    V = np.zeros((m * p, rank), dtype=int)  # 16 x 32
    W = np.zeros((p * n, rank), dtype=int)  # 16 x 32
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i * m + j
                # V index: j * p + k
                # W index: k * n + i
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
