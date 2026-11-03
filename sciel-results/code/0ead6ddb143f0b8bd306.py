import time
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS  # 3, 3, 7
    
    # Rank 27 decomposition
    # Split B into B1 (3x3), B2 (3x3), B3 (3x1)
    # C = A*B1 + A*B2 + A*B3
    # Each term is a schoolbook multiplication.
    # Rank of A*B1 is 9
    # Rank of A*B2 is 9
    # Rank of A*B3 is 9
    # Total rank = 27
    
    rank = 27
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Block 1: B1 (cols 0-2 of B)
    for i in range(n):
        for j in range(m):
            for k in range(3):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    # Block 2: B2 (cols 3-5 of B)
    for i in range(n):
        for j in range(m):
            for k in range(3):
                U[i * m + j][col] = 1
                V[j * p + (3 + k)][col] = 1
                W[(3 + k) * n + i][col] = 1
                col += 1
                
    # Block 3: B3 (col 6 of B)
    for i in range(n):
        for j in range(m):
            U[i * m + j][col] = 1
            V[j * p + 6][col] = 1
            W[6 * n + i][col] = 1
            col += 1
            
    if time.time() > deadline - 10:
        return {"U": U, "V": V, "W": W}
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
