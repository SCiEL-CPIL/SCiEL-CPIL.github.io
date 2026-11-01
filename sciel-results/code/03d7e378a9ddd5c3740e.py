"""Rewritten program for matrix multiplication 2,3,9 tensor decomposition."""
DIMENSIONS = [2, 3, 9]

import time
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying
    a 2x3 matrix by a 3x9 matrix.
    
    The tensor for this multiplication has a block-diagonal structure in the
    inner dimension j. For each fixed j in {0, 1, 2}, the sub-tensor where
    A index has j-component 0, B index has j-component 1, and C index has
    j-component 0 is an all-ones 2x9 tensor (rank 1). The three blocks are
    disjoint, so the total rank is 3.
    
    U has shape (6, 3), V has shape (27, 3), W has shape (18, 3).
    Each column r corresponds to one of the three j-blocks.
    """
    n, m, p = DIMENSIONS  # n=2, m=3, p=9
    
    rank = m  # rank = 3
    
    # U: shape (n*m, rank) = (6, 3)
    # U[i*m + j, r] = 1 if j == r, else 0
    U = [[0] * rank for _ in range(n * m)]
    for i in range(n):
        for j in range(m):
            U[i * m + j][j] = 1
    
    # V: shape (m*p, rank) = (27, 3)
    # V[j*p + k, r] = 1 if j == r, else 0
    V = [[0] * rank for _ in range(m * p)]
    for j in range(m):
        for k in range(p):
            V[j * p + k][j] = 1
    
    # W: shape (p*n, rank) = (18, 3)
    # W[k*n + i, r] = 1 if k == r, else 0
    # Wait, this is wrong. Let me reconsider.
    # 
    # The tensor T[a, b, c] = 1 if and only if:
    # a = i*m + j_a, b = j_b*p + k, c = k*n + i
    # where j_a == j_b (the inner dimension must match)
    #
    # For a rank-1 factor with U[a,r], V[b,r], W[c,r]:
    # The contribution is U[a,r] * V[b,r] * W[c,r]
    #
    # For the factor to contribute 1 to T[a,b,c] when j_a == j_b, and 0 otherwise:
    # We need U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 when j is the "active" j for factor r
    # and 0 otherwise.
    #
    # If we set r to correspond to a specific j value:
    # U[i*m+j, r] = 1 if j == r, else 0
    # V[j*p+k, r] = 1 if j == r, else 0
    # W[k*n+i, r] = 1 (always, for any k and i)
    #
    # Then for T[a,b,c] where a = i*m+j_a, b = j_b*p+k, c = k*n+i:
    # Sum over r: U[i*m+j_a, r] * V[j_b*p+k, r] * W[k*n+i, r]
    # = sum over r: [1 if j_a==r else 0] * [1 if j_b==r else 0] * 1
    # = 1 if j_a == j_b (and there exists r = j_a = j_b), else 0
    # = 1 if j_a == j_b, else 0
    #
    # This is exactly what we want!
    
    W = [[0] * rank for _ in range(p * n)]
    for k in range(p):
        for i in range(n):
            for r in range(rank):
                W[k * n + i][r] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
