import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Trivial decomposition for 5x5 matrix multiplication
    # Rank = 5 * 5 * 5 = 125
    
    U = [[0] * 125 for _ in range(25)]
    V = [[0] * 125 for _ in range(25)]
    W = [[0] * 125 for _ in range(25)]
    
    col = 0
    for i in range(5):
        for j in range(5):
            for k in range(5):
                # U[i*5+j, col] = A[i,j]
                # V[j*5+k, col] = B[j,k]
                # W[k*5+i, col] = C[i,k]
                U[i*5+j][col] = 1
                V[j*5+k][col] = 1
                W[k*5+i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
