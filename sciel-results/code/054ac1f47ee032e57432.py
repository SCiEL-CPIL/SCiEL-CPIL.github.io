"""
Program to find the rank-minimal exact tensor decomposition for 2x3 by 3x4 matrix multiplication.
The dimensions are n=2, m=3, p=4.
The known minimal rank for M_{2,3,4} is 24.
"""

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p  # 24
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U maps input A[i,j] to component col
                U[i * m + j][col] = 1
                # V maps input B[j,k] to component col
                V[j * p + k][col] = 1
                # W maps component col to output C[i,k]
                # Output index is k*n + i according to the convention W[k*n+i, r]
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
