import numpy as np
import time

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    # T[i, k, j] = 1 if we're computing C[i,k] from A[i,j] B[j,k]
    # Actually the tensor: T[i,k,j] corresponds to the product A[i,j]*B[j,k] contributing to C[i,k]
    # U indexed by (i,j): row i*m+j
    # V indexed by (j,k): row j*p+k
    # W indexed by (k,i): row k*n+i
    # Decomposition: T[i,k,j] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Build target tensor as a 3D array T[i, k, j]
    T = np.zeros((n, p, m))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, k, j] = 1.0
    
    # Schoolbook rank
    rank_sb = n*m*p
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best = (rank_sb, U_sb, V_sb, W_sb)
    
    # Try to find lower rank via optimization
    deadline = time.time() + 100  # safety margin
    for r in range(14, rank_sb):  # try 14, 15, 16, 17
        if time.time() > deadline:
            break
        found = search_rank(T, n, m, p, r, deadline)
        if found is not None:
            U, V, W = found
            best = (r, U, V, W)
            break  # found lowest r, stop
    
    # Convert to lists
    U, V, W = best[1], best[2], best[3]
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
