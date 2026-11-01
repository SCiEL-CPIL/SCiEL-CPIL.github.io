import numpy as np

def construct():
    n, m, p = 2, 2, 13
    nm = n * m
    mp = m * p
    pn = p * n
    
    R = 52
    U = np.zeros((nm, R), dtype=float)
    V = np.zeros((mp, R), dtype=float)
    W = np.zeros((pn, R), dtype=float)
    
    col = 0
    for k in range(p):
        for i in range(n):
            for j in range(m):
                # U index for A[i, j]
                u_idx = i * m + j
                # V index for B[j, k]
                v_idx = j * p + k
                # W index for C[i, k]
                w_idx = i * p + k
                
                U[u_idx, col] = 1
                V[v_idx, col] = 1
                W[w_idx, col] = 1
                col += 1
                
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
