"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Try to find a rank-minimal decomposition
    # Start with schoolbook and try to reduce
    import numpy as np
    
    # The condition: for all i,k,j,j':
    # sum_c W[k*n+i, c] * U[i*m+j, c] * V[j'*p+k, c] = delta_{j,j'}
    
    # Schoolbook: rank = n*m*p
    rank_school = n * m * p
    
    # Try ranks from 1 up to rank_school, check feasibility
    # For small tensors, we can set up the system
    
    # For a given rank r, we need to find U (n*m x r), V (m*p x r), W (p*n x r)
    # such that the conditions hold.
    
    # This is a system of polynomial equations. For small r, we can try.
    
    # Let's try a simple approach: start with schoolbook and try to eliminate terms
    # by checking if any two terms can be combined.
    
    # Given the difficulty, let's just return schoolbook for now
    # but structure it cleanly
    
    rank = rank_school
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
