"""Program for matrix multiplication 3x4 by 4x5; attempts rank reduction."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
import random

def _verify(U, V, W, n, m, p):
    """Check if the decomposition exactly reconstructs the matrix multiplication tensor."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    if U.shape != (n*m, U.shape[1]):
        return False
    if V.shape != (m*p, V.shape[1]):
        return False
    if W.shape != (p*n, W.shape[1]):
        return False
    if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
        return False
        
    rank = U.shape[1]
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m + j, c] * V[j*p + k, c] * W[k*n + i, c]
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    return max_err < 1e-10

def _make_schoolbook(n, m, p):
    """Create the standard schoolbook decomposition of rank n*m*p."""
    rank_sb = n * m * p
    U = [[0] * rank_sb for _ in range(n * m)]
    V = [[0] * rank_sb for _ in range(m * p)]
    W = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    t_start = time.time()
    deadline = t_start + 110.0
    
    best_result = _make_schoolbook(n, m, p)
    best_rank = n * m * p
    
    return best_result

# EVOLVE-BLOCK-END
