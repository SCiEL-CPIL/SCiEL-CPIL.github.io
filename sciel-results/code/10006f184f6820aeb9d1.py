import time
import numpy as np

DIMENSIONS = [2, 3, 7]

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix by a 3 by 7 matrix.
    """
    start_time = time.time()
    deadline = start_time + 145.0
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 42
    
    # Schoolbook decomposition
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    best = {"U": U, "V": V, "W": W}
    best_rank = rank_schoolbook
    
    return best
