"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns the schoolbook tensor decomposition for matrix multiplication.
    The rank is n*m*p.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices as lists of lists of zeros
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Each rank-1 term corresponds to the product A[i,j] * B[j,k] contributing to C[i,k]
                # In the flattened indexing:
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i  (Note: W[k*n+i] matches the trace convention W[k*n+i,r] in prompt)
                
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
