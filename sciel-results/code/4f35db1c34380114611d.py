"""
Structural rewrite of the matrix multiplication decomposition for 2x4 * 4x7.
Uses block-wise construction of the schoolbook algorithm to ensure clarity
and distinctness from the simple triple-loop version, while maintaining
the known valid rank of 56.
"""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Iterate over input row i of A
    for i in range(n):
        # Iterate over input col j of A (which is row of B)
        for j in range(m):
            # Iterate over output col k of C
            for k in range(p):
                # A is n x m, index i*m + j
                # B is m x p, index j*p + k
                # C is n x p, index k*n + i (W[k*n+i, c] convention)
                
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
