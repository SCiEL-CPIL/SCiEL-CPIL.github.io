import numpy as np
import time
import random

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Build the target tensor
    # T[i, j, k] = 1 if the term e_i \otimes f_j \otimes g_k is in the tensor
    # The tensor is sum_{i,j,k} e_{i,j} \otimes f_{j,k} \otimes g_{k,i}
    # So T[i*m+j, j*p+k, k*n+i] = 1
    # We want to find U, V, W such that sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = T[i*m+j, j*p+k, k*n+i]
    
    # The target tensor is sparse.
    # We can check the decomposition by verifying the equation.
    
    # For now, return the schoolbook decomposition.
    # A stronger algorithm would search for a lower rank.
    # Given the time limit, we can try a simple search.
    
    # Let's try to find a decomposition with rank r < rank_schoolbook.
    # We can use a randomized search.
    
    # But for now, let's just return the schoolbook.
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
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
