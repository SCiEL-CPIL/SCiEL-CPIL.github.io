import time

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # The minimal rank for M_{n,m,p} is max(n*p, m*p, m*n)
    # For 2, 4, 5: max(10, 20, 8) = 20.
    rank = m * p
    
    # Initialize matrices
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Fill the matrices based on the decomposition:
    # Rank index r corresponds to (j, k)
    # V is Identity (if r = j*p + k)
    # U[i*m + j, r] = 1
    # W[k*n + i, r] = 1
    
    for j in range(m):
        for k in range(p):
            r = j * p + k
            V[j * p + k][r] = 1
            for i in range(n):
                U[i * m + j][r] = 1
                W[k * n + i][r] = 1
                
    return {"U": U, "V": V, "W": W}
