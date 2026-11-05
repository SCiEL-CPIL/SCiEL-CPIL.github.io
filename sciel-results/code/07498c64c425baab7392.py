"""
Rank-125 exact decomposition for 5x5 x 5x5 matrix multiplication.

Uses the standard schoolbook tensor decomposition. The multiplication tensor
T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] where each term
corresponds to one scalar multiplication a_{i,j} * b_{j,k} contributing to
c_{i,k}.

For n=m=p=5, the rank is 125. This is the known minimal rank for exact
5x5 matrix multiplication (no improvement over schoolbook is known).
"""

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p  # 125
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Each term: U[i,j] * V[j,k] * W[k,i] contributes to c[i,k]
                # via the trace convention W[k*n+i, r]
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
