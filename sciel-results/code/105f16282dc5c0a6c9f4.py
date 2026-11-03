"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import numpy as np
from scipy.linalg import expm

def construct():
    n, m, p = DIMENSIONS
    # Start with the schoolbook rank-48 decomposition
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = i * m * p + j * p + k
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
    
    # Apply random linear transformations to see if we can find a lower-rank
    # representation by exploiting structure (this is a heuristic; the actual
    # rank should not change under invertible linear transformations, but
    # sometimes numerical issues or specific structures allow apparent reductions)
    # However, a more direct approach: use a known rank-reduction for matrix multiplication.
    # For 2x3x8, the schoolbook rank is 48. The true minimal rank might be lower.
    # Let's try a construction based on Strassen-like methods for smaller blocks.
    
    # Actually, let's try a different strategy: use a random decomposition of rank 40
    # and check if it works (it won't), so let's stick to a valid decomposition.
    # We can try to use a known algorithm that gives rank < 48 for 2x3x8.
    # One such algorithm is based on the fact that 2x3x2 can be done in rank 11
    # (or similar), and we can compose that with the 8 dimension.
    
    # Let's use a valid rank-48 decomposition but with a different structure.
    # For now, return the schoolbook decomposition as it is guaranteed to be correct.
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
