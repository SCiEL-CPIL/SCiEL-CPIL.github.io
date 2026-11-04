"""
Optimized program for matrix multiplication 2x3 * 3x9.
Uses a block-diagonal decomposition strategy where the computation for each row of the A matrix
is treated as an independent 1x3 * 3x9 multiplication.
"""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Rank is n * m * p for this block-diagonal structure
    total_rank = n * m * p
    
    # Use lists of zeros
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Fill the matrices
    # We iterate through all combinations of i, j, k
    # Each combination corresponds to one term in the decomposition
    term_idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Set the 1s in the factor matrices
                U[i * m + j][term_idx] = 1
                V[j * p + k][term_idx] = 1
                W[k * n + i][term_idx] = 1
                term_idx += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
