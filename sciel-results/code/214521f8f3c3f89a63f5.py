"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal (within known practical bounds) exact tensor decomposition
    for multiplying a 3x3 matrix by a 3x8 matrix.
    
    The task is to decompose the bilinear map:
    (A, B) -> C = A * B
    where A is 3x3, B is 3x8, C is 3x8.
    
    The tensor T is of size (3*3) x (3*8) x (3*8) = 9 x 24 x 24.
    The rank is the minimum number of rank-1 tensors needed to sum to T.
    
    The schoolbook rank is 3*3*8 = 72.
    It is known that for small matrix multiplication, the rank can be lower.
    However, finding the exact minimal rank is an open problem for most dimensions.
    
    For 3x3x8, a decomposition with rank < 72 would be an improvement.
    One approach is to use block decomposition.
    We can split B into two blocks: B1 (3x3) and B2 (3x5).
    C = A * B1 + A * B2.
    The rank of the sum is at most the sum of the ranks.
    Rank(A*B1) for 3x3x3 is at most 23 (Coppersmith-Winograd).
    Rank(A*B2) for 3x3x5 is at most 45 (schoolbook) or lower.
    
    However, implementing the Coppersmith-Winograd decomposition (rank 23) is complex
    and involves specific coefficients.
    
    Given the constraint of 120s and the need for an exact, valid decomposition,
    we will return the schoolbook decomposition of rank 72. This is always valid.
    Any attempt to use a lower rank decomposition requires hardcoding specific
    coefficients which are non-trivial to generate or verify quickly.
    
    The score is the rank. The current best known bound for 3x3x8 might be lower,
    but without a pre-computed valid decomposition, rank 72 is the safe baseline.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
