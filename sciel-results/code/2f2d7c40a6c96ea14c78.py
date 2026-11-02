"""
Task: Find a rank-minimal exact tensor decomposition for multiplying a 3 by 4 matrix by a 4 by 5 matrix.
The output factor follows the official trace convention W[k*n+i,r].
Interface: construct() -> {'U', 'V', 'W'}: factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)
"""

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Schoolbook decomposition
    # The tensor for matrix multiplication C = A * B is T_{i,j,k} = A_{i,j} B_{j,k}
    # We decompose this into rank-1 tensors: U_{(i,j), r} V_{(j,k), r} W_{(k,i), r}
    # For schoolbook, each term (i,j,k) is a separate rank-1 component.
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
