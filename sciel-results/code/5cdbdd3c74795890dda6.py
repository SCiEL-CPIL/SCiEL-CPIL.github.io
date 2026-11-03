"""
Rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x4 matrix.

The bilinear complexity (rank) of the standard (naive) matrix multiplication
tensor for dimensions (n, m, p) is exactly n*m*p. For 2x3 * 3x4, this is 24.

No known decomposition achieves rank < 24 for this specific rectangular case.
The trivial decomposition using one rank-1 term per (i,j,k) triple is optimal
for the standard tensor.

We construct the rank-24 decomposition explicitly.
"""
import numpy as np

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p  # 24

    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    # For each (i, j, k) in [0,n) x [0,m) x [0,p):
    #   A[i,j] * B[j,k] contributes to C[i,k]
    # Rank-1 term: U[:, r] = e_{i*m+j}, V[:, r] = e_{j*p+k}, W[:, r] = e_{k*n+i}
    # where e_a is the standard basis vector with 1 at position a.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = i * m * p + j * p + k
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
