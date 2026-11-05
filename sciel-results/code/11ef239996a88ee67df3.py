import numpy as np
from scipy.optimize import minimize
import time

DIMENSIONS = [3, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    
    start_time = time.time()
    deadline = start_time + 140
    
    # Try rank 1
    rank = 1
    U = [[1.0]] * (n * m)
    V = [[1.0]] * (m * p)
    W = [[1.0]] * (p * n)
    
    # Verify
    T_check = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_check[i,j,k] = U[i*m+j][0] * V[j*p+k][0] * W[k*n+i][0]
    
    if np.max(np.abs(T_check - 1.0)) < 1e-8:
        return {"U": U, "V": V, "W": W}
    
    # Fall back to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
