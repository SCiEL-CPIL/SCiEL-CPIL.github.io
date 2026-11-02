import time

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    # Schoolbook decomposition of rank n*m*p: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c].
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    # Using lists of lists for simplicity and compatibility with the expected interface
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Fill the matrices
    # The tensor T[i, j, k] = 1 if and only if it corresponds to the multiplication A[i, j] * B[j, k]
    # The decomposition is sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # For schoolbook, each term is independent.
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
