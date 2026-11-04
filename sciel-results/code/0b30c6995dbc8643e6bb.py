import time
import numpy as np

DIMENSIONS = [4, 5, 5]
DEADLINE = 150.0
SAFETY_MARGIN = 5.0

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    # n=4, m=5, p=5
    # U shape: (n*m, rank) = (20, rank)
    # V shape: (m*p, rank) = (25, rank)
    # W shape: (p*n, rank) = (20, rank)
    
    U_terms = []
    V_terms = []
    W_terms = []
    
    # Trivial decomposition of rank N*M*P
    # For each i in 0..n-1, j in 0..m-1, k in 0..p-1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = [0] * (n * m)
                v = [0] * (m * p)
                w = [0] * (p * n)
                
                # u index: i*m + j
                u[i * m + j] = 1
                
                # v index: j*p + k
                v[j * p + k] = 1
                
                # w index: k*n + i
                w[k * n + i] = 1
                
                U_terms.append(u)
                V_terms.append(v)
                W_terms.append(w)
                
    rank = len(U_terms)
    U = np.array(U_terms).T
    V = np.array(V_terms).T
    W = np.array(W_terms).T
    
    return {"U": U, "V": V, "W": W}
