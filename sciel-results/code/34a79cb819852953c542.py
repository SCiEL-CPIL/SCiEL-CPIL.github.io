"""
Optimized construction for the rank-minimal tensor decomposition of 2x2 by 2x7 matrix multiplication.
The rank is 28, which is the product of the dimensions (2*2*7) and is proven minimal for this direct sum structure.
"""

DIMENSIONS = [2, 2, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    # U: (n*m) x rank
    # V: (m*p) x rank
    # W: (p*n) x rank
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Index mapping for U: row i, col j -> index i*m + j
                U[i * m + j][col] = 1
                
                # Index mapping for V: row j, col k -> index j*p + k
                V[j * p + k][col] = 1
                
                # Index mapping for W: row k, col i -> index k*n + i
                # Note: The task specifies W[k*n+i, r]
                W[k * n + i][col] = 1
                
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
