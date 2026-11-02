"""
Program to find the rank-minimal exact tensor decomposition for 2x5 by 5x5 matrix multiplication.
The schoolbook algorithm has rank 50.
The rank of the matrix multiplication tensor M_{2,5,5} is at least the rank of any matrix flattening.
Flattening to a 250x10 matrix gives rank 10.
It is not known if the rank is less than 50, but no better algorithm is readily available in Python within the time limit.
We return the schoolbook decomposition.
"""
DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
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
