import numpy as np

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # Use a known rank-reduction for rectangular matrix multiplication
    # For 4x5x6, we can achieve rank 119 by exploiting one linear dependency
    rank = 119
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    # Standard schoolbook terms for all (i, j, k) except one redundant term
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Skip one term to reduce rank by 1 (if there's a dependency)
                if i == 0 and j == 0 and k == 0:
                    continue
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    # We need to account for the skipped term (0,0,0)
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + ... 
    # The skipped term was for i=0, j=0, k=0, which contributes to C[0,0]
    # We can't simply skip it; we need a valid decomposition
    # This approach is incorrect; let's use the full schoolbook rank
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
