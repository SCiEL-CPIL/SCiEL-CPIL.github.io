import numpy as np

DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 2 * p = 22
    rank = 2 * p
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: Handle j=0 term (A[i,0] * B[0,k])
    # Columns 0 to p-1
    for k in range(p):
        r = k
        # U: rows i*m + 0 are 1, i*m + 1 are 0
        for i in range(n):
            U[i * m + 0][r] = 1
            U[i * m + 1][r] = 0
            
        # V: rows 0*p + k is 1, others 0
        for j in range(m):
            for kk in range(p):
                if j == 0 and kk == k:
                    V[j * p + kk][r] = 1
                else:
                    V[j * p + kk][r] = 0
                    
        # W: rows k*n + i are 1
        for i in range(n):
            W[k * n + i][r] = 1
            
    # Part 2: Handle j=1 term (A[i,1] * B[1,k])
    # Columns p to 2p-1
    for k in range(p):
        r = p + k
        # U: rows i*m + 0 are 0, i*m + 1 are 1
        for i in range(n):
            U[i * m + 0][r] = 0
            U[i * m + 1][r] = 1
            
        # V: rows 1*p + k is 1, others 0
        for j in range(m):
            for kk in range(p):
                if j == 1 and kk == k:
                    V[j * p + kk][r] = 1
                else:
                    V[j * p + kk][r] = 0
                    
        # W: rows k*n + i are 1
        for i in range(n):
            W[k * n + i][r] = 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
