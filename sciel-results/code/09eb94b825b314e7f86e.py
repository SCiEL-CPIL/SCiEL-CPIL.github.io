"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # The rank of M_{n,m,p} is n*m*p when p is such that no non-trivial
    # decomposition exists, specifically for M_{n,m,1} rank is n*m.
    # M_{n,m,p} is isomorphic to p copies of M_{n,m,1}.
    # Thus rank = p * n * m.
    rank = n * m * p
    
    # Initialize matrices with zeros
    # Using lists of lists for compatibility and simplicity
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U is indexed by [i*m + j, column]
                U[i * m + j][column] = 1
                # V is indexed by [j*p + k, column]
                V[j * p + k][column] = 1
                # W is indexed by [k*n + i, column]
                # Convention: W[k*n+i, r]
                W[k * n + i][column] = 1
                column += 1
                
                # Check time occasionally (not really needed for rank 63, but good practice)
                if time.time() > deadline:
                    return {"U": U, "V": V, "W": W}
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
