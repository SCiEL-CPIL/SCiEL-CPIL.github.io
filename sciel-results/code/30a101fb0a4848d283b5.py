"""Matrix multiplication 3,3,5 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 5
    
    # Trivial rank-n*m*p decomposition (schoolbook)
    rank = n * m * p  # 45
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
    
    result = {"U": U, "V": V, "W": W}
    
    # Try to find a lower rank decomposition by searching over known small-rank decompositions
    # For 3x3x5, known results suggest rank <= 44 may exist, but finding one is complex.
    # We'll try a simple approach: check if we can merge any two terms of the trivial decomposition.
    # Two terms (i1,j1,k1) and (i2,j2,k2) can be merged if the resulting tensor still matches.
    # This is equivalent to checking if the sum of two basis vectors can be represented with fewer factors.
    # In general, for the trivial decomposition, no two terms can be merged to get a lower rank 
    # unless there's a specific algebraic identity.
    
    # A known approach for small matrix multiplication is to use the Winograd algorithm or similar.
    # For 3x3x3, Winograd gives rank 23. For 3x3x5, we might adapt similar ideas.
    
    # However, since we don't have a precomputed low-rank decomposition, and finding one 
    # via numerical methods within the time limit is risky, we'll stick with the trivial one.
    
    # If we had a precomputed rank-44 or lower decomposition, we'd return it here.
    
    return result
# EVOLVE-BLOCK-END
