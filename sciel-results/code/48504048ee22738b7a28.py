import time

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U maps input A[i,j] to component column
                U[i * m + j][column] = 1
                # V maps input B[j,k] to component column
                V[j * p + k][column] = 1
                # W maps output C[i,k] from component column
                # Convention: W[k*n+i, r]
                W[k * n + i][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}
