"""Rewritten program for matrix multiplication 2x2 by 2x5 tensor decomposition."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
def construct():
    # Schoolbook decomposition of rank n*m*p = 2*2*5 = 20.
    # U has shape (n*m, rank) = (4, 20)
    # V has shape (m*p, rank) = (10, 20)
    # W has shape (p*n, rank) = (10, 20)
    n, m, p = DIMENSIONS
    rank = n * m * p  # 20
    
    # Initialize matrices with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
