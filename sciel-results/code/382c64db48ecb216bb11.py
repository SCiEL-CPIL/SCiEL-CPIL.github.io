import time
import numpy as np

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    
    # Block decomposition:
    # Split m=5 into 4+1, p=5 into 4+1.
    # A is 2x5, B is 5x5, C is 2x5.
    # A = [A1 (2x4), A2 (2x1)]
    # B = [[B11 (4x4), B12 (4x1)], [B21 (1x4), B22 (1x1)]]
    # C = A1*B11 + A1*B12 + A2*B21 + A2*B22
    
    # We use schoolbook for each block, summing the ranks.
    # Rank 1: A1*B11 (2x4x4) -> 16 terms
    # Rank 2: A1*B12 (2x4x1) -> 8 terms
    # Rank 3: A2*B21 (2x1x4) -> 8 terms
    # Rank 4: A2*B22 (2x1x1) -> 2 terms
    # Total rank = 34
    
    rank = 34
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Block 1: A1 (2x4) * B11 (4x4) -> C11 (2x4)
    # i in 0..1, j in 0..3, k in 0..3
    for i in range(2):
        for j in range(4):
            for k in range(4):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                U[idx_u][col] = 1
                V[idx_v][col] = 1
                W[idx_w][col] = 1
                col += 1
                
    # Block 2: A1 (2x4) * B12 (4x1) -> C12 (2x1)
    # i in 0..1, j in 0..3, k=4
    for i in range(2):
        for j in range(4):
            k = 4
            idx_u = i * m + j
            idx_v = j * p + k
            idx_w = k * n + i
            U[idx_u][col] = 1
            V[idx_v][col] = 1
            W[idx_w][col] = 1
            col += 1
            
    # Block 3: A2 (2x1) * B21 (1x4) -> C21 (2x4)
    # i in 0..1, j=4, k in 0..3
    for i in range(2):
        j = 4
        for k in range(4):
            idx_u = i * m + j
            idx_v = j * p + k
            idx_w = k * n + i
            U[idx_u][col] = 1
            V[idx_v][col] = 1
            W[idx_w][col] = 1
            col += 1
            
    # Block 4: A2 (2x1) * B22 (1x1) -> C22 (2x1)
    # i in 0..1, j=4, k=4
    for i in range(2):
        j = 4
        k = 4
        idx_u = i * m + j
        idx_v = j * p + k
        idx_w = k * n + i
        U[idx_u][col] = 1
        V[idx_v][col] = 1
        W[idx_w][col] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
