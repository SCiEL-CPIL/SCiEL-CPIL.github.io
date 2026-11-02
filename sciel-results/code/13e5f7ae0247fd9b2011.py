import numpy as np
from itertools import product

def construct():
    n, m, p = 2, 4, 6
    
    # Rank 24 decomposition
    rank = 24
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for j in range(m):
        for k in range(p):
            # u_r has 1 at (0, j) and (1, j)
            # Index in U: i*m + j
            U[0 * m + j][column] = 1
            U[1 * m + j][column] = 1
            
            # v_r has 1 at (j, k)
            # Index in V: j*p + k
            V[j * p + k][column] = 1
            
            # w_r has 1 at (k, 0) and (k, 1)
            # Index in W: k*n + i
            W[k * n + 0][column] = 1
            W[k * n + 1][column] = 1
            
            column += 1
            
    return {"U": U, "V": V, "W": W}
