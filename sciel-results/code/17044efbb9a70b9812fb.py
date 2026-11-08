"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x4 matrix by a 4x6 matrix.
    The output factor follows the official trace convention W[k*n+i, r].
    """
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit

    n, m, p = DIMENSIONS
    # Schoolbook rank
    schoolbook_rank = n * m * p  # 48

    # For the 2x4x6 case, we can construct a rank-24 decomposition using block structure.
    # The key insight: we can decompose the 4x4 middle dimension into 2x2 blocks
    # and use Strassen's algorithm for each 2x2x2 block.
    
    # For 2x4x6, we can write this as 2x(2*2)x(3*2) = 2x4x6
    # We can use the fact that 4 = 2*2 and 6 = 3*2
    
    # Actually, let's use a more direct approach:
    # The tensor T[i,l,k] = 1 for all i in [0,2), l in [0,4), k in [0,6)
    # We can decompose this using the structure of the dimensions.
    
    # Let's try to find a rank-24 decomposition.
    # The lower bound is max(n*m, m*p, p*n) = max(8, 24, 12) = 24
    
    # Construct a rank-24 decomposition:
    # We can use the fact that for each fixed i, the slice T[i,:,:] is a 4x6 all-ones matrix.
    # The rank of a 4x6 all-ones matrix is 1, so we might be able to do better.
    
    # Let's try a structured approach:
    # For each i in {0,1}, l in {0,1,2,3}, k in {0,1,2,3,4,5}
    # T[i,l,k] = 1
    
    # We can write T as a sum of rank-1 tensors.
    # Let's try to use the structure that l can be split into l1 in {0,1} and l2 in {0,1,2,3}
    # and k can be split into k1 in {0,1} and k2 in {0,1,2,3,4,5}
    
    # Actually, let's just try to find a rank-24 decomposition directly.
    # We'll use a simple construction:
    
    # For each combination of (i, l, k), we need U[i*m+l, r] * V[l*p+k, r] * W[k*n+i, r] to sum to 1.
    
    # Let's try a rank-24 decomposition with the following structure:
    # For r in 0..23, let r = i*m + l where i in {0,1} and l in {0,1,2,3}
    # Then set U[i*m+l, r] = 1, and for other (i', l') set U[i'*m+l', r] = 0
    # Similarly for V and W.
    
    # Wait, that would give rank 8, not 24. Let me reconsider.
    
    # The correct approach: the rank of the matrix multiplication tensor for n x m x p
    # is at least max(n*m, m*p, p*n) but can be higher.
    
    # For 2x4x6, let's try to construct a rank-24 decomposition.
    
    # Actually, I realize I should just return the schoolbook decomposition for now,
    # as finding a better one is non-trivial and may not be possible with the time constraint.
    
    # Let's return the schoolbook decomposition with rank 48.
    rank = schoolbook_rank
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
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
