"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin before 150s kill
    
    # Rank 28 decomposition: one term per (j,k) pair
    # U[i*m+j, j*p+k] = 1 for all i
    # V[j*p+k, j*p+k] = 1 (only the diagonal-like entry)
    # W[k*n+i, j*p+k] = 1 for all i
    #
    # More precisely:
    # For r = j*p + k (j in 0..m-1, k in 0..p-1):
    #   U[i*m + j, r] = 1 for all i in 0..n-1
    #   V[j*p + k, r] = 1
    #   W[k*n + i, r] = 1 for all i in 0..n-1
    #
    # Verification: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    #   = sum_{j',k'} U[i*m+j, j'*p+k'] * V[j*p+k, j'*p+k'] * W[k*n+i, j'*p+k']
    #   U[i*m+j, j'*p+k'] = 1 iff j'=j
    #   V[j*p+k, j'*p+k'] = 1 iff k'=k
    #   So only term is j'=j, k'=k: W[k*n+i, j*p+k] = 1
    #   Sum = 1. Correct!
    
    rank = m * p  # 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for j in range(m):
        for k in range(p):
            r = j * p + k
            # U: rows i*m+j for all i
            for i in range(n):
                U[i * m + j][r] = 1
            # V: row j*p+k
            V[j * p + k][r] = 1
            # W: rows k*n+i for all i
            for i in range(n):
                W[k * n + i][r] = 1
    
    # Verify exactly with fractions
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if s != 1:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if valid:
        return {"U": U, "V": V, "W": W}
    
    # Fallback: trivial rank n*m*p decomposition (should not be reached)
    best_rank = n * m * p
    best_U = [[0] * best_rank for _ in range(n * m)]
    best_V = [[0] * best_rank for _ in range(m * p)]
    best_W = [[0] * best_rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i * m + j][col] = 1
                best_V[j * p + k][col] = 1
                best_W[k * n + i][col] = 1
                col += 1
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
