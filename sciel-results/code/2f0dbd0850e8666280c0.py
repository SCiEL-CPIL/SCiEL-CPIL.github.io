"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Exact tensor decomposition for multiplying a 4x5 matrix by a 5x5 matrix.
    Uses a direct construction with rank optimization.
    """
    n, m, p = DIMENSIONS  # n=4, m=5, p=5
    # U: (n*m) x rank, V: (m*p) x rank, W: (p*n) x rank
    # Convention: C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # summed over j implicitly through the tensor structure
    
    # Start with schoolbook: rank = n*m*p = 100
    # Each term corresponds to a specific (i,j,k) triple
    terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = {(i, j): 1}
                v = {(j, k): 1}
                w = {(k, i): 1}
                terms.append((u, v, w))
    
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, (u, v, w) in enumerate(terms):
        for (i, j), val in u.items():
            U[i * m + j][r] += val
        for (j, k), val in v.items():
            V[j * p + k][r] += val
        for (k, i), val in w.items():
            W[k * n + i][r] += val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
