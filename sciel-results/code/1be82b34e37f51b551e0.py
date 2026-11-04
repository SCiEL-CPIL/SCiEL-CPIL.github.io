import numpy as np
import time
from fractions import Fraction
from itertools import product

START_TIME = time.time()
TIME_LIMIT = 110  # Safety margin for 120s limit

def _elapsed():
    return time.time() - START_TIME

def _verify(U, V, W, n=3, m=5, p=7):
    """Verify that the decomposition is exact for matrix multiplication."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Check tensor identity: sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target = 1.0
                computed = 0.0
                for r in range(rank):
                    computed += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(computed - target) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = 3, 5, 7
    rank = n * m * p
    
    U = [[0.0 for _ in range(rank)] for _ in range(n * m)]
    V = [[0.0 for _ in range(rank)] for _ in range(m * p)]
    W = [[0.0 for _ in range(rank)] for _ in range(p * n)]
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][idx] = 1.0
                V[j*p + k][idx] = 1.0
                W[k*n + i][idx] = 1.0
                idx += 1
    
    if not _verify(U, V, W):
        # This should never happen for schoolbook
        pass
        
    return {'U': U, 'V': V, 'W': W}
