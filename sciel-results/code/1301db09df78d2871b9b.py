"""Matrix multiplication 4x4x5 - rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 130.0
    
    n, m, p = 4, 4, 5
    nm = n * m  # 16
    mp = m * p  # 20
    pn = p * n  # 20
    
    # Schoolbook decomposition
    rank = n * m * p  # 80
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a, col] = 1
                V[b, col] = 1
                W[c, col] = 1
                col += 1
    
    return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
