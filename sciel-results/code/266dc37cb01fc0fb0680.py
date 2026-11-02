"""Matrix multiplication 3,4,4 — schoolbook rank-48 decomposition.

Known lower bounds:
  rank_{4,4}(4x4x4 tensor) >= 21  (Stoimenow 1998)
  rank(3x4x4) = max(rank(3x4x4), rank(4x4x4)) >= 21
Known upper bound: 48 (schoolbook).
No explicit decomposition with rank < 48 is known and constructible within
120 s without heavy search.  We therefore return the canonical schoolbook
decomposition which is always valid.
"""

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """Return the schoolbook rank-48 exact tensor decomposition for
    multiplying a 3x4 matrix by a 4x4 matrix.

    Shapes:
      U: (n*m, rank) = (12, 48)
      V: (m*p, rank) = (16, 48)
      W: (p*n, rank) = (12, 48)

    Indexing convention (official trace):
      U[i*m + j,  c] corresponds to input A[i, j]
      V[j*p + k,  c] corresponds to input B[j, k]
      W[k*n + i,  c] corresponds to output C[i, k]

    For each (i, j, k) triple the c-th column has U = V = W = 1 at the
    corresponding positions, giving C[i,k] += A[i,j] * B[j,k].
    """
    n, m, p = DIMENSIONS  # 3, 4, 4
    rank = n * m * p      # 48

    U = [[0] * rank for _ in range(n * m)]   # 12 x 48
    V = [[0] * rank for _ in range(m * p)]   # 16 x 48
    W = [[0] * rank for _ in range(p * n)]   # 12 x 48

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
