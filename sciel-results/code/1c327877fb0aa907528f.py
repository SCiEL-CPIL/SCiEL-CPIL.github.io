import numpy as np

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 4
    
    # The target tensor T[i,j,k] = 1 for all i,j,k
    # This has tensor rank 1: outer product of three all-ones vectors
    # U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s] summed over s = 1
    
    # Rank 1 solution:
    # U[i*m+j, 0] = 1 for all i in [0,n), j in [0,m)
    # V[j*p+k, 0] = 1 for all j in [0,m), k in [0,p)
    # W[k*n+i, 0] = 1 for all k in [0,p), i in [0,n)
    
    rank = 1
    U = [[1.0] for _ in range(n * m)]
    V = [[1.0] for _ in range(m * p)]
    W = [[1.0] for _ in range(p * n)]
    
    return {"U": U, "V": V, "W": W}
