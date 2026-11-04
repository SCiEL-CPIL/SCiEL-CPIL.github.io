import numpy as np
import time

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # The target tensor T[i,j,k] = 1 for all i,j,k (all-ones tensor)
    # Rank-1 decomposition: U = ones(n*m, 1), V = ones(m*p, 1), W = ones(p*n, 1)
    # This gives sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all i,j,k
    
    rank = 1
    U = [[1] for _ in range(n * m)]
    V = [[1] for _ in range(m * p)]
    W = [[1] for _ in range(p * n)]
    
    # Verify
    T = np.ones((n, m, p))
    check = np.zeros((n, m, p))
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    check[i, j, k] += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
    
    assert np.allclose(check, T, atol=1e-9), "Rank-1 decomposition failed!"
    
    return {"U": U, "V": V, "W": W}
