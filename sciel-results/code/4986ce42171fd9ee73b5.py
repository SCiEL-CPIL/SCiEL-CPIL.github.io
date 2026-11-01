import numpy as np
from itertools import product

def construct():
    n, m, p = 2, 3, 9
    
    # The matrix multiplication tensor for (n x m) * (m x p) -> (p x n) [trace convention]
    # can be decomposed into p independent instances of the (n x m) * (m x 1) -> (1 x n) tensor.
    # Each instance computes one column of the output matrix.
    # The rank of the (n, m, 1) tensor is n*m (since each output element depends on a unique
    # product of one element from A and one from B, and there are n*m such products).
    # Thus, the total rank is p * n * m.
    # For n=2, m=3, p=9, rank = 9 * 2 * 3 = 54.
    
    rank = n * m * p
    
    # Initialize factor matrices as lists of lists
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Fill in the decomposition
    # The tensor is a direct sum of p copies of the (n, m, 1) tensor.
    # For each column k of the output (0 to p-1):
    #   The (n, m, 1) tensor has rank n*m.
    #   For each i in 0..n-1 and j in 0..m-1:
    #     U[i*m + j][col] = 1
    #     V[j*p + k][col] = 1
    #     W[k*n + i][col] = 1  (trace convention: W[k*n + i, r])
    
    col = 0
    for k in range(p):
        for i in range(n):
            for j in range(m):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
