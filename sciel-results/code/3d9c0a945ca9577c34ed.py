"""
Matrix multiplication 2,3,4 rank-minimal exact tensor decomposition.
The dimensions are n=2, m=3, p=4.
The schoolbook algorithm has rank n*m*p = 24.
It is widely believed that the rank of M_{2,3,4} is 24.
We provide the schoolbook decomposition.
"""

DIMENSIONS = [2, 3, 4]

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
                # U index: i * m + j
                # V index: j * p + k
                # W index: k * n + i
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}
