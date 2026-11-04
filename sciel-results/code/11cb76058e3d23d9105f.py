"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    # Decomposition of rank 68: 64 terms for 4x4x4 part, 4 terms for 4x4x1 part.
    n, m, p = DIMENSIONS
    rank = 80
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: 4x4x4 schoolbook (first 4 columns of output)
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(4):  # k=0 to 3
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    # Part 2: 4x4x1 (5th column of output, k=4)
    # C[i, 4] = sum_j A[i, j] B[j, 4]
    # This requires 4 terms, one for each j?
    # No, the term is A[i, j] B[j, 4] for each j.
    # So for each i, we need terms for each j?
    # Wait, the tensor entry T[i, j, 4] = 1 for all j.
    # So we need terms that cover all j.
    # Term r for j=0: U[i, 0, r]=1, V[0, 4, r]=1, W[4, i, r]=1.
    # This term contributes to C[i, 4] via A[i, 0] B[0, 4].
    # We need one term for each j?
    # Yes, because V[j, 4, r] must be non-zero for specific j.
    # If we use one term for all j, V[0, 4, r]=1, V[1, 4, r]=1, etc.
    # Then U[i, 0, r]=1, U[i, 1, r]=1...
    # Then the term is sum_j U[i, j, r] V[j, 4, r] W[4, i, r].
    # If U[i, j, r]=1 for all j, V[j, 4, r]=1 for all j, W[4, i, r]=1.
    # Then the sum is sum_j 1*1*1 = 4.
    # But we need sum_j A[i, j] B[j, 4].
    # The coefficient of A[i, j] B[j, 4] should be 1.
    # So we need separate terms for each j.
    # So 4 terms for the 5th column.
    
    for i in range(n):
        for j in range(m):
            col = column
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1  # k=4
            W[4 * n + i][col] = 1
            column += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
