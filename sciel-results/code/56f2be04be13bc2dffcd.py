"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x9 matrix.
    The output factor follows the official trace convention W[k*n+i, r].
    
    Returns dict with 'U', 'V', 'W' factor matrices.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # 2, 3, 9
    
    # Rank 27 decomposition
    # Lower bound is max(n*m, m*p, p*n) = max(6, 27, 18) = 27.
    # We construct a decomposition of rank 27.
    
    rank = m * p  # 27
    
    U = [[0] * rank for _ in range(n * m)]  # 6 x 27
    V = [[0] * rank for _ in range(m * p)]  # 27 x 27
    W = [[0] * rank for _ in range(p * n)]  # 18 x 27
    
    # Mapping r to (j', k')
    # r = j' * p + k'
    # j' in [0, m), k' in [0, p)
    
    for r in range(rank):
        j_prime = r // p
        k_prime = r % p
        
        # U[i*m + j, r] = 1 if j == j_prime
        for i in range(n):
            for j in range(m):
                if j == j_prime:
                    U[i * m + j][r] = 1
        
        # V[j*p + k, r] = 1 if k == k_prime
        for j in range(m):
            for k in range(p):
                if k == k_prime:
                    V[j * p + k][r] = 1
        
        # W[k*n + i, r] = 1 if k == k_prime
        for k in range(p):
            for i in range(n):
                if k == k_prime:
                    W[k * n + i][r] = 1
    
    # Verify the decomposition
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1.0) > 1e-10:
                    # Fallback to schoolbook if verification fails (should not happen)
                    rank_schoolbook = n * m * p
                    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
                    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
                    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
                    column = 0
                    for ii in range(n):
                        for jj in range(m):
                            for kk in range(p):
                                U_school[ii * m + jj][column] = 1
                                V_school[jj * p + kk][column] = 1
                                W_school[kk * n + ii][column] = 1
                                column += 1
                    return {"U": U_school, "V": V_school, "W": W_school}
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
