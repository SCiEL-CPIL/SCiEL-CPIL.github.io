def construct():
    import numpy as np
    import time

    n, m, p = 3, 4, 4
    
    # We construct a standard bilinear decomposition with rank n*m*p = 48.
    # This ensures correctness and validity.
    rank = n * m * p
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                U[i * m + j, r] = 1.0
                # V index: j*p + k
                V[j * p + k, r] = 1.0
                # W index: k*n + i
                W[k * n + i, r] = 1.0
                r += 1
                
    return {'U': U, 'V': V, 'W': W}
