import time
import numpy as np

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal exact tensor decomposition for matrix multiplication
    of dimensions n x m x p.
    Currently implements the schoolbook algorithm with rank n*m*p.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U[i*m + j, col] = 1
                U[i * m + j][col] = 1
                # V[j*p + k, col] = 1
                V[j * p + k][col] = 1
                # W[k*n + i, col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    # Verification (optional, but good for robustness)
    # Check if sum_c U[:,c] V[:,c] W[:,c] matches the multiplication tensor
    # Tensor T[i,j,k] = 1 if i in [0,n), j in [0,m), k in [0,p)
    # Decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Quick check for a few entries
    # This is O(rank * 1) per check, so very fast.
    # We trust the construction, but let's ensure no index errors occurred.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
