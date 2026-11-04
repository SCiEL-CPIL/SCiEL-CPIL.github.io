import time
import sys

DIMENSIONS = [3, 3, 5]

def construct():
    start_time = time.time()
    deadline = 145  # 150s limit, 5s margin
    
    n, m, p = DIMENSIONS
    
    # We will use a split decomposition:
    # Part 1: k=0,1,2 (3x3x3) using Schoolbook (Rank 9)
    # Part 2: k=3,4 (3x3x2) using Schoolbook (Rank 18)
    # Total Rank = 27
    
    rank = 27
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: k in 0, 1, 2
    for k in range(3):
        for i in range(n):
            for j in range(m):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}

    # Part 2: k in 3, 4
    for k in range(3, 5):
        for i in range(n):
            for j in range(m):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
