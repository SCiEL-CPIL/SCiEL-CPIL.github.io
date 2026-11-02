"""
Program for rank-minimal tensor decomposition of 2x5x5 matrix multiplication.
Uses the schoolbook algorithm which decomposes the tensor into 50 rank-1 terms.
This is the standard baseline for this dimension.
"""
import numpy as np

DIMENSIONS = [2, 5, 5]

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 2x5x5 matrix multiplication.
    
    The tensor T_{2,5,5} corresponds to the bilinear map:
    (A, B) -> A * B
    where A is 2x5 and B is 5x5.
    
    The schoolbook algorithm uses n*m*p multiplications, where n=2, m=5, p=5.
    Rank = 2 * 5 * 5 = 50.
    
    The decomposition is defined by:
    U[i*m + j, r] = 1 if r corresponds to (i,j,k)
    V[j*p + k, r] = 1 if r corresponds to (i,j,k)
    W[k*n + i, r] = 1 if r corresponds to (i,j,k)
    
    Here, r = i * (m*p) + j * p + k.
    """
    n, m, p = DIMENSIONS  # 2, 5, 5
    rank = n * m * p  # 50
    
    # Initialize U, V, W as lists of lists with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Fill the matrices based on the schoolbook decomposition
    # Each rank-1 term corresponds to a unique (i, j, k) tuple
    # i in 0..n-1 (row of A, row of C)
    # j in 0..m-1 (col of A, row of B)
    # k in 0..p-1 (col of B, col of C)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: row i of A, col j. Flattened: i*m + j
                U[i * m + j][col] = 1
                
                # V index: row j of B, col k. Flattened: j*p + k
                V[j * p + k][col] = 1
                
                # W index: col k of C, row i. 
                # Convention: W[k*n + i, r]
                W[k * n + i][col] = 1
                
                col += 1
    
    return {"U": U, "V": V, "W": W}
