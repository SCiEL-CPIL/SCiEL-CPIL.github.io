"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
import time
import sys

def construct():
    """
    Constructs a rank-30 exact tensor decomposition for 2x3 by 3x5 matrix multiplication.
    
    The schoolbook algorithm has rank n*m*p = 2*3*5 = 30.
    Each term corresponds to a unique triple (i, j, k) where:
    i in [0, n), j in [0, m), k in [0, p)
    
    The tensor entry T[(i_A, j_A), (j_B, k_B), (i_C, k_C)] is 1 if:
    i_A == i_C and j_A == j_B and k_B == k_C.
    
    We decompose this as:
    T = sum_{j=0}^{m-1} sum_{i=0}^{n-1} sum_{k=0}^{p-1} 
        U_{(i, j)} \otimes V_{(j, k)} \otimes W_{(i, k)}
    
    Where:
    U_{(i, j)} is a matrix of size n x m with 1 at (i, j) and 0 elsewhere.
    V_{(j, k)} is a matrix of size m x p with 1 at (j, k) and 0 elsewhere.
    W_{(i, k)} is a matrix of size p x n with 1 at (k, i) and 0 elsewhere.
    
    Verification:
    Sum over r (indexing i, j, k):
    U_{r, i_A, j_A} = 1 if i_A == i and j_A == j, else 0.
    V_{r, j_B, k_B} = 1 if j_B == j and k_B == k, else 0.
    W_{r, i_C, k_C} = 1 if i_C == i and k_C == k, else 0.
    
    Product is 1 only if i_A=i, j_A=j, j_B=j, k_B=k, i_C=i, k_C=k.
    This implies i_A=i_C=i, j_A=j_B=j, k_B=k_C=k.
    This matches the target tensor definition.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                U[i * m + j][col] = 1
                # V index: j*p + k
                V[j * p + k][col] = 1
                # W index: k*n + i (following W[k*n+i, r] convention)
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
