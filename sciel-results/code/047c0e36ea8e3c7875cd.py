"""Direct decomposition for multiplying a 3x3 matrix by a 3x8 matrix.

The multiplication tensor for C = A B has entries
    T_{i,j,k} = A_{i,j} * B_{j,k},   for i in [n), j in [m), k in [p].

A rank-1 decomposition term for indices (i,j,k) is given by:
    U_{a, r} = 1   if a = i*m + j   else 0
    V_{b, r} = 1   if b = j*p + k   else 0
    W_{c, r} = 1   if c = k*n + i   else 0

where r is the term index. This gives the schoolbook rank n*m*p = 72.

We output this exact decomposition.
"""
import time

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard
    t0 = time.time()
    
    n, m, p = DIMENSIONS
    rank = n * m * p  # 3*3*8 = 72
    
    # Allocate matrices
    U = [[0] * rank for _ in range(n * m)]   # 9 x 72
    V = [[0] * rank for _ in range(m * p)]   # 24 x 72
    W = [[0] * rank for _ in range(p * n)]   # 24 x 72
    
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
