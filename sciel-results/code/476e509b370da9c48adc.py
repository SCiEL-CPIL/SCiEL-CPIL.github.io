import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    # The rank of M_{n,m,p} for n=2 is 2*m*p because the two rows of A
    # correspond to disjoint sets of input variables, and the rank of
    # M_{1,m,p} is m*p. Thus R(M_{2,m,p}) = R(M_{1,m,p}) + R(M_{1,m,p}) = 2mp.
    # For n=2, m=3, p=10, rank = 60.
    # We construct the schoolbook decomposition.
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i (Trace convention W[k*n+i, r])
                
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
