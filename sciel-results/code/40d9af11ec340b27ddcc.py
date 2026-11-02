import numpy as np

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    # Split p=7 into 3, 3, 1
    p1, p2, p3 = 3, 3, 1
    
    # Rank for 3x3x3 schoolbook is 27
    # Rank for 3x3x1 schoolbook is 3
    rank1 = n * m * p1
    rank2 = n * m * p2
    rank3 = n * m * p3
    
    total_rank = rank1 + rank2 + rank3
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col_offset = 0
    
    # Block 1: cols 0 to 2
    for i in range(n):
        for j in range(m):
            for k in range(p1):
                U[i * m + j][col_offset] = 1
                V[j * p + k][col_offset] = 1
                W[k * n + i][col_offset] = 1
                col_offset += 1
                
    # Block 2: cols 3 to 5
    for i in range(n):
        for j in range(m):
            for k in range(p2):
                U[i * m + j][col_offset] = 1
                V[j * p + (p1 + k)][col_offset] = 1
                W[(p1 + k) * n + i][col_offset] = 1
                col_offset += 1
                
    # Block 3: col 6
    for i in range(n):
        for j in range(m):
            # k is 0 relative to block, 6 absolute
            abs_k = p1 + p2
            U[i * m + j][col_offset] = 1
            V[j * p + abs_k][col_offset] = 1
            W[abs_k * n + i][col_offset] = 1
            col_offset += 1
            
    return {"U": U, "V": V, "W": W}
