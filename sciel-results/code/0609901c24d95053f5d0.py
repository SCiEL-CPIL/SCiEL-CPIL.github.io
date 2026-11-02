import numpy as np

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    # Strategy: Decompose into p independent (n, m, 1) tensor instances.
    # Rank of (n, m, 1) is n*m.
    # Total rank is p * n * m.
    rank = n * m * p
    
    # Initialize matrices with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    # Iterate over each of the p output columns
    for k in range(p):
        # For each column k, we perform (n, m, 1) multiplication
        # which requires n*m rank-1 terms
        for i in range(n):
            for j in range(m):
                # U index: (i, j) -> i*m + j
                u_idx = i * m + j
                # V index: (j, k) -> j*p + k
                v_idx = j * p + k
                # W index: (k, i) -> k*n + i (Trace convention W[k*n+i, r])
                w_idx = k * n + i
                
                U[u_idx][r] = 1
                V[v_idx][r] = 1
                W[w_idx][r] = 1
                
                r += 1
                
    return {"U": U, "V": V, "W": W}
