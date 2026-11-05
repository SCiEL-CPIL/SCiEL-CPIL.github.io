import numpy as np

def construct():
    n, m, p = 3, 4, 7
    
    # The naive rank is n * m * p
    rank = n * m * p
    
    # We will construct the U, V, W matrices.
    # U is (n*m, rank)
    # V is (m*p, rank)
    # W is (p*n, rank)
    
    # Using lists of lists to build the matrices, then converting to numpy or keeping as lists.
    # The interface expects lists of lists (based on .tolist() in seed).
    
    U = [[0.0 for _ in range(rank)] for _ in range(n * m)]
    V = [[0.0 for _ in range(rank)] for _ in range(m * p)]
    W = [[0.0 for _ in range(rank)] for _ in range(p * n)]
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i (Trace convention W[k*n+i, r])
                
                U[i * m + j][idx] = 1.0
                V[j * p + k][idx] = 1.0
                W[k * n + i][idx] = 1.0
                idx += 1
                
    return {"U": U, "V": V, "W": W}
