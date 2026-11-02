import time
import numpy as np

DIMENSIONS = [3, 3, 6]

def _check_half_integer(arr, tol=1e-9):
    doubled = arr * 2.0
    rounded = np.round(doubled)
    if np.max(np.abs(doubled - rounded)) < tol:
        return rounded / 2.0
    return None

def _verify_decomposition_fast(U, V, W, n, m, p):
    rank = len(U[0])
    U_arr = np.array(U, dtype=float).reshape(n, m, rank)
    V_arr = np.array(V, dtype=float).reshape(m, p, rank)
    W_arr = np.array(W, dtype=float).reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kir->ijk', U_arr, V_arr, W_arr, optimize=True)
    return np.allclose(T, np.ones((n, m, p)), atol=1e-9)

def construct():
    start_time = time.time()
    deadline = start_time + 110
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition
    # Rank = n * m * p = 54
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
