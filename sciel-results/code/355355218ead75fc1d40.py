"""Matrix multiplication 3,4,5 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3 by 4 matrix by a 4 by 5 matrix.
    The output factor follows the official trace convention W[k*n+i,r].
    
    Returns dict with 'U', 'V', 'W' keys containing factor matrices.
    """
    n, m, p = DIMENSIONS
    # n=3 rows of A, m=4 cols of A / rows of B, p=5 cols of B
    # A is n x m = 3 x 4
    # B is m x p = 4 x 5
    # C is n x p = 3 x 5
    
    # Schoolbook decomposition has rank n*m*p = 60
    # We'll try to find a better decomposition using block structure.
    
    # Strategy: Split the middle dimension m=4 into blocks.
    # If we split m=4 into m1=2, m2=2, then:
    # A = [A1 | A2] where A1 is 3x2, A2 is 3x2
    # B = [B1; B2] where B1 is 2x5, B2 is 2x5
    # C = A1*B1 + A2*B2
    
    # Each product A1*B1 is (3x2)*(2x5), schoolbook rank = 3*2*5 = 30
    # Total schoolbook = 60
    
    # Can we do better for (3x2)*(2x5)?
    # The tensor M_{3,2,5} has rank at least max(3*2, 2*5, 5*3) = max(6,10,15) = 15
    # Schoolbook is 30.
    
    # For now, let's use the schoolbook decomposition which is guaranteed correct.
    # rank = 60
    
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
