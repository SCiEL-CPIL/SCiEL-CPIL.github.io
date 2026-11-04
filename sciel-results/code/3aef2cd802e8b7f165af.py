import numpy as np

def construct():
    """
    Construct a tensor decomposition for 2x4 * 4x8 matrix multiplication.
    
    Uses schoolbook (naive) multiplication with rank 64.
    This is guaranteed to be correct, though not rank-minimal.
    """
    
    n, m, p = 2, 4, 8
    rank = n * m * p  # 2 * 4 * 8 = 64
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U[i*m + j, r] = 1
                # V[j*p + k, r] = 1
                # W[k*n + i, r] = 1
                U[i * m + j, r] = 1
                V[j * p + k, r] = 1
                W[k * n + i, r] = 1
                r += 1
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
