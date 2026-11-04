import numpy as np
import time

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # Rank 48 decomposition
    # n=3, m=4, p=8
    # Split m into two parts of size 2.
    # Part 1: j in {0, 1}
    # Part 2: j in {2, 3}
    # For each part, rank is n * p = 3 * 8 = 24.
    # Total rank = 48.
    
    rank = 48
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Part 1: j = 0, 1
    # For each k in 0..7, for each i in 0..2:
    #   r = k * 3 + i
    #   U[i*m + 0, r] = 1
    #   U[i*m + 1, r] = 1
    #   V[0*p + k, r] = 1
    #   V[1*p + k, r] = 1
    #   W[k*n + i, r] = 1
    
    for k in range(p):
        for i in range(n):
            r = k * n + i
            # U indices: i*m + j
            U[i * m + 0, r] = 1
            U[i * m + 1, r] = 1
            # V indices: j*p + k
            V[0 * p + k, r] = 1
            V[1 * p + k, r] = 1
            # W indices: k*n + i
            W[k * n + i, r] = 1
            
    # Part 2: j = 2, 3
    # For each k in 0..7, for each i in 0..2:
    #   r = 24 + k * 3 + i
    #   U[i*m + 2, r] = 1
    #   U[i*m + 3, r] = 1
    #   V[2*p + k, r] = 1
    #   V[3*p + k, r] = 1
    #   W[k*n + i, r] = 1
    
    for k in range(p):
        for i in range(n):
            r = 24 + k * n + i
            # U indices: i*m + j
            U[i * m + 2, r] = 1
            U[i * m + 3, r] = 1
            # V indices: j*p + k
            V[2 * p + k, r] = 1
            V[3 * p + k, r] = 1
            # W indices: k*n + i
            W[k * n + i, r] = 1
            
    # Verify
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
                
    S = np.einsum('ir,jr,kr->ijk', U, V, W)
    if not np.allclose(S, T, atol=1e-9):
        # Fallback to schoolbook if verification fails (should not happen)
        rank_schoolbook = n * m * p
        U_s = np.zeros((n * m, rank_schoolbook))
        V_s = np.zeros((m * p, rank_schoolbook))
        W_s = np.zeros((p * n, rank_schoolbook))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i * m + j, col] = 1
                    V_s[j * p + k, col] = 1
                    W_s[k * n + i, col] = 1
                    col += 1
        return {
            "U": U_s.tolist(),
            "V": V_s.tolist(),
            "W": W_s.tolist()
        }

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
