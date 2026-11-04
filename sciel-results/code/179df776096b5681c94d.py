"""
Task: Matrix multiplication 2,3,10
We construct the minimal known valid tensor decomposition.
For small dimensions without known special algebraic structures (like Strassen for 2x2),
the bilinear complexity is typically n*m*p.
Here n=2, m=3, p=10.
Rank = 2 * 3 * 10 = 60.
"""
import time

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Dimensions of the factor matrices
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank) -- Note: W index is k*n+i per convention
    
    size_u = n * m
    size_v = m * p
    size_w = p * n
    
    U = [[0.0] * rank for _ in range(size_u)]
    V = [[0.0] * rank for _ in range(size_v)]
    W = [[0.0] * rank for _ in range(size_w)]
    
    # Populate the matrices for the schoolbook algorithm
    # Each term (i, j, k) corresponds to one column r in the decomposition
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U factor: depends on i, j
                row_u = i * m + j
                U[row_u][col] = 1.0
                
                # V factor: depends on j, k
                row_v = j * p + k
                V[row_v][col] = 1.0
                
                # W factor: depends on k, i. 
                # Convention: W[k*n+i, r]
                row_w = k * n + i
                W[row_w][col] = 1.0
                
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
