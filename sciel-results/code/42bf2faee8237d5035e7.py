import numpy as np
from fractions import Fraction
import time

def construct():
    n, m, p = 2, 4, 7
    rank = 4
    
    U = [[0] * rank for _ in range(n * m)]  # (8, 4)
    V = [[0] * rank for _ in range(m * p)]  # (28, 4)
    W = [[0] * rank for _ in range(p * n)]  # (14, 4)
    
    for r in range(rank):
        # U: A[i, r] coefficient
        U[0 * m + r][r] = 1  # i=0, j=r
        U[1 * m + r][r] = 1  # i=1, j=r
        
        # V: B[r, k] coefficient for all k
        for k in range(p):
            V[r * p + k][r] = 1
        
        # W: output coefficient for all k, i
        for k in range(p):
            for i in range(n):
                W[k * n + i][r] = 1
    
    return {"U": U, "V": V, "W": W}
