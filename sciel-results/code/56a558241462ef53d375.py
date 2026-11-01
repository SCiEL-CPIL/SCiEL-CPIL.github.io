"""
Exact tensor decomposition for 3x3 @ 3x4 matrix multiplication.

We use the standard schoolbook decomposition.  For A (3x3) and B (3x4),
C = A @ B has entries c_{i,k} = sum_{j=0}^{2} a_{i,j} b_{j,k}.

The multiplication tensor T has indices:
  a = i * m + j   (row of A, column of A)  -> 0..8
  b = j * p + k   (row of B, column of B)  -> 0..11
  c = k * n + i   (row of C, column of C)  -> 0..11

Each triple (i, j, k) gives one rank-1 term with support at (a, b, c).
Since the mapping (i,j,k) -> (a,b,c) is injective, all 36 terms are
linearly independent and the rank is exactly n*m*p = 36.
"""

def construct():
    n, m, p = 3, 3, 4
    nm = n * m      # 9
    mp = m * p      # 12
    pn = p * n      # 12
    rank = n * m * p  # 36

    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]

    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * m * p + j * p + k
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a][r] = 1
                V[b][r] = 1
                W[c][r] = 1

    return {"U": U, "V": V, "W": W}
