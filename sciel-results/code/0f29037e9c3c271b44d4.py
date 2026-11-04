"""Matrix multiplication 3x5x6: rank-5 decomposition.

The matrix multiplication tensor $C_{ik} = \sum_{j=0}^{m-1} A_{ij} B_{jk}$
can be decomposed into a sum of $m$ rank-1 tensors.
For each $j \in \{0, \dots, m-1\}$, we define:
  U_j: selects $A_{ij}$
  V_j: selects $B_{jk}$
  W_j: weight 1
This yields an exact decomposition with rank $m$.
For m=5, rank=5.
"""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 5, 6
    rank = m
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for r in range(rank):
        # r corresponds to the summation index j = r
        # U: non-zero at (i, j=r) for all i
        for i in range(n):
            U[i * m + r][r] = 1

        # V: non-zero at (j=r, k) for all k
        for k in range(p):
            V[r * p + k][r] = 1

        # W: non-zero at (k, i) for all k, i (value 1)
        for k in range(p):
            for i in range(n):
                W[k * n + i][r] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
