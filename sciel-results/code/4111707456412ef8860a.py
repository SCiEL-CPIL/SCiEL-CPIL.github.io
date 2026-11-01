import numpy as np

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Trivial decomposition:
    # Each term corresponds to a single (i, j, k) index triple.
    # U[i*m + j, t] = 1 if t maps to (i, j, k)
    # V[j*p + k, t] = 1 if t maps to (i, j, k)
    # W[k*n + i, t] = 1 if t maps to (i, j, k)
    
    t = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx, t] = 1
                V[v_idx, t] = 1
                W[w_idx, t] = 1
                t += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
