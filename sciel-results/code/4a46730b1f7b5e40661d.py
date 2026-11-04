<<<<<<< SEARCH
col = offset
U[(i_start+0)*m + (j_start+0)][col] += 1
U[(i_start+1)*m + (j_start+1)][col] += 1
V[(j_start+0)*p + (k_start+0)][col] += 1
V[(j_start+1)*p + (k_start+1)][col] += 1
W[(k_start+0)*n + (i_start+0)][col] += 1
W[(k_start+1)*n + (i_start+1)][col] += 1
=======
import numpy as np
import time

def construct():
    m, n, p = 2, 4, 6
    rank = m * n * p  # 48, trivial decomposition
    
    U = np.zeros((m * n, rank), dtype=np.float64)
    V = np.zeros((n * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    for i in range(m):
        for j in range(n):
            for k in range(p):
                r = i * n * p + j * p + k
                U[i * n + j, r] = 1.0
                V[j * p + k, r] = 1.0
                W[k * n + i, r] = 1.0
    
    return {'U': U, 'V': V, 'W': W}
>>>>>>> REPLACE
