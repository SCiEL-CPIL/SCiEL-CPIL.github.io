import time

DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # The tensor for multiplying a (n x m) matrix by a (m x p) matrix
    # is the direct sum of p copies of the tensor for (n x m) by (m x 1).
    # The rank of M_{n,m,1} is n*m.
    # Here n=2, m=2, so rank of M_{2,2,1} is 4.
    # Total rank = p * (n * m) = 9 * 4 = 36.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i * m + j
                # V index: j * p + k
                # W index: k * n + i
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
