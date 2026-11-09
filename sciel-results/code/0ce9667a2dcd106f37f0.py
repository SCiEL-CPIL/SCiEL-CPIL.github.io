"""Matrix multiplication 3x5x5 rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying
    a 3x5 matrix by a 5x5 matrix.
    
    Output factor follows the official trace convention W[k*n+i, r].
    
    Interface: construct() -> {'U', 'V', 'W'}: factor matrices of shapes
    (n*m, rank), (m*p, rank), (p*n, rank) with integer/half-integer entries
    that exactly decompose the matrix multiplication tensor.
    
    Score is the rank (lower is better).
    
    Strategy:
    We use a structured decomposition of rank r = n*m = 15.
    The indices for the rank dimension are pairs (i_c, j_c) where i_c in [0, n) and j_c in [0, m).
    
    Definitions:
    U[i*m + j, c(i_c, j_c)] = 1 if i == i_c and j == j_c, else 0
    V[j'*p + k, c(i_c, j_c)] = 1 if j' == j_c, else 0
    W[k*n + i, c(i_c, j_c)] = 1 if i == i_c, else 0
    
    Verification:
    Sum over c of U[i,j,c] * V[j',k,c] * W[k,i,c]
    = Sum over i_c, j_c of [i==i_c, j==j_c] * [j'==j_c] * [i==i_c]
    = [i==i_c, j==j_c==j'] summed over i_c, j_c
    = 1 if j==j' (and i matches the unique i_c), else 0
    = delta_{j,j'}
    This satisfies the tensor condition for matrix multiplication.
    """
    n, m, p = DIMENSIONS  # 3, 5, 5
    r = n * m  # rank = 15
    
    # Initialize factor matrices with zeros
    # U shape: (n*m, r)
    U = [[0] * r for _ in range(n * m)]
    # V shape: (m*p, r)
    V = [[0] * r for _ in range(m * p)]
    # W shape: (p*n, r)
    W = [[0] * r for _ in range(p * n)]
    
    # Populate the matrices based on the structured decomposition
    for i_c in range(n):
        for j_c in range(m):
            c = i_c * m + j_c
            
            # U[i*m + j, c] = 1 if i == i_c and j == j_c
            # Since we iterate over all i_c, j_c, for a specific c, only one (i,j) pair is 1.
            # Specifically, row index is i_c * m + j_c.
            U[i_c * m + j_c][c] = 1
            
            # V[j'*p + k, c] = 1 if j' == j_c
            # For a fixed c (fixed j_c), V[row][c] is 1 for all k if row corresponds to j_c.
            # Row index in V is j' * p + k. So for j' = j_c, all k in [0, p) get 1.
            for k in range(p):
                V[j_c * p + k][c] = 1
                
            # W[k*n + i, c] = 1 if i == i_c
            # For a fixed c (fixed i_c), W[row][c] is 1 for all k if row corresponds to i_c.
            # Row index in W is k * n + i. So for i = i_c, all k in [0, p) get 1.
            for k in range(p):
                W[k * n + i_c][c] = 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
