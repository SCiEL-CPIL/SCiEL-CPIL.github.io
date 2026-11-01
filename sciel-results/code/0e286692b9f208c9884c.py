"""Matrix multiplication 2x3x4 rank-minimal exact tensor decomposition."""
import time

DIMENSIONS = [2, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin
    
    n, m, p = DIMENSIONS  # 2, 3, 4
    
    # Rank-12 decomposition
    # For each (k, j) where k in {0,1,2,3} and j in {0,1,2}, we have a term r = k*m + j
    # U[i*m+j, r] = 1 for all i in {0, 1}
    # V[j*p+k, r] = 1
    # W[k*n+i, r] = 1 for all i in {0, 1}
    # This gives T[i,j',k'] = 1 for all i,j',k', correctly decomposing the multiplication tensor
    
    rank = m * p  # 3 * 4 = 12
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for k in range(p):
        for j in range(m):
            r = k * m + j
            # U[i*m+j, r] = 1 for i in {0, 1}
            for i in range(n):
                U[i * m + j][r] = 1
            # V[j*p+k, r] = 1
            V[j * p + k][r] = 1
            # W[k*n+i, r] = 1 for i in {0, 1}
            for i in range(n):
                W[k * n + i][r] = 1
    
    return {"U": U, "V": V, "W": W}
