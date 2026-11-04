import numpy as np

def construct():
    n, m, p = 4, 5, 6
    R = n * m * p
    U = np.zeros((n*m, R))
    V = np.zeros((m*p, R))
    W = np.zeros((p*n, R))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                U[i*m + j, col] = 1.0
                # V index: j*p + k
                V[j*p + k, col] = 1.0
                # W index: k*n + i (Trace convention W[k*n+i, r])
                W[k*n + i, col] = 1.0
                col += 1
                
    return {'U': U, 'V': V, 'W': W}
