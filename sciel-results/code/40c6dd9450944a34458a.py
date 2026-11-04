"""Matrix multiplication 2x3x7 rank-minimal exact tensor decomposition.

Uses the schoolbook (direct) method with rank = n*m*p = 42.
Each (i,j,k) triple gets its own rank-1 component.

Conventions:
  U: (n*m, rank)  -- U[i*m + j, c]
  V: (m*p, rank)  -- V[j*p + k, c]
  W: (p*n, rank)  -- W[k*n + i, c]
"""

def construct():
    n, m, p = 2, 3, 7
    rank = n * m * p  # 42

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

    return {"U": U, "V": V, "W": W}
