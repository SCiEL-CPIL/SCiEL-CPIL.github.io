"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    rank = n * m * p  # Schoolbook rank 30
    
    # Construct the schoolbook decomposition
    # U[i*m+j, r] = 1 if r corresponds to (i,j,k)
    # V[j*p+k, r] = 1 if r corresponds to (i,j,k)
    # W[k*n+i, r] = 1 if r corresponds to (i,j,k)
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, r] = 1.0
                V[j*p+k, r] = 1.0
                W[k*n+i, r] = 1.0
                r += 1
    
    # Return as lists of floats (which are integers, hence valid half-integers)
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
