"""
Matrix multiplication 4x4 * 4x5: rank-minimal exact tensor decomposition.

We construct a rank-80 schoolbook decomposition, which is guaranteed correct.
The tensor is:
  C[i,k] = sum_j A[i,j] * B[j,k]
where A is 4x4, B is 4x5, C is 4x5.

Indexing convention:
  U has shape (n*m, rank) = (16, rank), row index = i*m + j
  V has shape (m*p, rank) = (20, rank), row index = j*p + k
  W has shape (p*n, rank) = (20, rank), row index = k*n + i

For each (i,j,k) triple, we create one rank-1 term:
  U[i*m+j][c] = 1
  V[j*p+k][c] = 1
  W[k*n+i][c] = 1

This gives C[i,k] = sum_c W[k*n+i][c] * sum_j U[i*m+j][c] * V[j*p+k][c]
                  = sum_j A[i,j]*B[j,k]  (when A,B are substituted)
"""

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # safety margin well under 120s

    n, m, p = DIMENSIONS  # 4, 4, 5
    rank = n * m * p      # 80

    # Initialize factor matrices as lists of lists (integer entries)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    # Guard: if somehow over deadline (shouldn't happen for O(80) work)
    if time.time() > deadline:
        pass

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
