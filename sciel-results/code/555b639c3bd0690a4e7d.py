"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    # Schoolbook decomposition of rank n*m*p.
    # Structure: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
                
    # Convert to lists as per interface expectation if strict, but numpy arrays are often accepted 
    # or the evaluator handles conversion. The prompt says "factor matrices of shapes ... with integer/half-integer entries".
    # The seed returned lists of lists. I will return lists of lists to be safe with the "exact interface" constraint 
    # unless the evaluator explicitly prefers arrays. The seed used lists.
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
