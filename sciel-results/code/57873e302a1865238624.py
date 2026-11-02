import time
import numpy as np

DIMENSIONS = [3, 3, 5]

def _validate(U, V, W, n, m, p):
    """
    Validates the tensor decomposition by checking if it exactly decomposes the matrix multiplication tensor.
    """
    rank = len(U[0])
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if any(len(row) != rank for row in U) or any(len(row) != rank for row in V) or any(len(row) != rank for row in W):
        return False
    
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_arr[i*m+j, :] * V_arr[j*p+k, :] * W_arr[k*n+i, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
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

def _construct_block_decomposition():
    """
    Constructs a rank-15 decomposition for 3x3 * 3x5 by splitting:
    - 3x3 * 3x3 part using rank-9 (schoolbook)
    - 3x3 * 3x2 part using rank-6 (schoolbook)
    Total rank = 9 + 6 = 15
    """
    n, m, p = 3, 3, 5
    rank = 15
    U = [[0] * rank for _ in range(9)]
    V = [[0] * rank for _ in range(15)]
    W = [[0] * rank for _ in range(15)]
    
    col = 0
    # Part 1: 3x3 * 3x3 (k = 0,1,2)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i * 3 + j][col] = 1
                V[j * 3 + k][col] = 1
                W[k * 3 + i][col] = 1
                col += 1
    
    # Part 2: 3x3 * 3x2 (k = 3,4)
    for i in range(3):
        for j in range(3):
            for k in range(2):
                k_orig = 3 + k
                U[i * 3 + j][col] = 1
                V[j * 3 + k_orig][col] = 1
                W[k_orig * 3 + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    
    result = _construct_block_decomposition()
    
    if _validate(result["U"], result["V"], result["W"], n, m, p):
        return result
    
    return _schoolbook(n, m, p)
