import numpy as np
import time

def construct():
    n, m, p = 3, 3, 4
    N = n * m  # 9
    M = m * p  # 12
    P = p * n  # 12
    
    rank = 12
    
    U = np.zeros((N, rank))
    V = np.zeros((M, rank))
    W = np.zeros((P, rank))
    
    # Decomposition based on 3x3x1 blocks for each column k
    # Global rank index g = k * 3 + r
    for k in range(p):
        for r in range(n):
            g = k * n + r
            
            # U index: u = i * m + j. We set i=r, j=r.
            u_idx = r * m + r
            U[u_idx, g] = 1.0
            
            # V index: v = j * p + k. We set j=r.
            v_idx = r * p + k
            V[v_idx, g] = 1.0
            
            # W index: w = k * n + i. We set i=r.
            w_idx = k * n + r
            W[w_idx, g] = 1.0
            
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
