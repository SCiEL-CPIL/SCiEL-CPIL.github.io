"""Matrix multiplication 4,4,7 rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 7]
DEADLINE = 120.0
START_TIME = time.time()

def _time_left():
    return DEADLINE - (time.time() - START_TIME)

def construct():
    """
    Returns a rank-112 schoolbook decomposition for 4x4 * 4x7 matrix multiplication.
    
    The tensor T_{i,j,k} = delta_{i,j} delta_{k,l} is decomposed as:
    sum_c U_{i*m+j, c} * V_{j*p+k, c} * W_{k*n+i, c}
    
    Schoolbook: for each (j,k), set U[i*m+j, c]=1, V[j*p+k, c]=1, W[k*n+i, c]=1
    """
    n, m, p = 4, 4, 7
    rank = n * m * p  # 112
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Convert to lists as expected by the interface
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
