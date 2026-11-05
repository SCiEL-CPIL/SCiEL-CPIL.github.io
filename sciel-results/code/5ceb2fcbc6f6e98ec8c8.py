import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = max(n * m, m * p, n * p)
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # V is Identity
    for idx in range(m * p):
        V[idx][idx] = 1
        
    # W is all 1s
    for i in range(p * n):
        for c in range(rank):
            W[i][c] = 1
            
    # U[i*m+j, c] = 1 if floor(c/p) == j
    for i in range(n):
        for j in range(m):
            row = i * m + j
            for c in range(rank):
                if c // p == j:
                    U[row][c] = 1
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
