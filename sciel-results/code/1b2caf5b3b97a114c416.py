import time
import numpy as np

def construct():
    n, m, p = 2, 3, 5
    deadline = time.time() + 140
    
    # Try ranks from 14 upward
    for rank in [14, 15, 16, 17, 18, 20, 25, 30]:
        if time.time() > deadline - 5:
            break
        # Try to find a decomposition
        result = try_find_decomposition(n, m, p, rank, deadline)
        if result is not None:
            return result
    
    # Fallback: schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def try_find_decomposition(n, m, p, rank, deadline):
    # ... implementation
    pass
