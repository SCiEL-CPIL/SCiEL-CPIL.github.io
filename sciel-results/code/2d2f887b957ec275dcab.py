import numpy as np

def construct():
    m = 4
    n = 4
    p = 6
    rank = m * n * p
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    r = 0
    for i in range(m):
        for j in range(n):
            for k in range(p):
                # U index: (i, j) -> i*n + j ? Or i*m + j?
                # Prompt says U shape (n*m, rank).
                # Usually row-major: i * n + j.
                # Let's assume i is row (0..m-1), j is col (0..n-1).
                # Index in U: i * n + j
                u_idx = i * n + j
                # V index: (j, k) -> j * p + k
                v_idx = j * p + k
                # W index: (k, i) -> k * n + i (per prompt W[k*n+i, r])
                w_idx = k * n + i
                
                U[u_idx, r] = 1.0
                V[v_idx, r] = 1.0
                W[w_idx, r] = 1.0
                r += 1
                
    return {'U': U, 'V': V, 'W': W}
