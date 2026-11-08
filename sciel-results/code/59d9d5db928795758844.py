"""Program for finding a rank-minimal exact tensor decomposition of matrix multiplication 3x4x6.

The task is to find U, V, W matrices of shapes (12, r), (24, r), (18, r) such that for all A (3x4) and B (4x6):
C = A @ B
where C[k*n+i, r_col] = sum over j of A[i, j] * B[j, k]

The tensor decomposition requires:
sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta_{j,j'} for appropriate indices

Actually, the standard decomposition is:
T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
where T is the matrix multiplication tensor with T[i,j,k] = 1 if the (i,j) entry of A and (j,k) entry of B contribute to (i,k) entry of C.

Wait, let me reconsider the convention. The output factor W has shape (p*n, rank) with indexing W[k*n+i, r].

For matrix multiplication C = A * B where A is 3x4, B is 4x6, C is 3x6:
C[i, k] = sum_j A[i, j] * B[j, k]

The tensor T has indices (i, j, k) where:
- i is row of A / row of C (0..n-1)
- j is column of A / row of B (0..m-1)  
- k is column of B / column of C (0..p-1)

T[i, j, k] = 1 (since A[i,j] * B[j,k] contributes to C[i,k])

The rank-r decomposition: T[i,j,k] = sum_{c=0}^{r-1} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]

The schoolbook solution uses r = n*m*p = 72 with each term being a single (i,j,k) triple.
"""

import time
import random
import numpy as np

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
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
