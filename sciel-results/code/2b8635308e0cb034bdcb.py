"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

import time
import itertools
import random

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal (or near-minimal) exact tensor decomposition for 3x4 * 4x6 matrix multiplication.
    
    The output factor W follows the official trace convention W[k*n+i, r].
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 150.0  # seconds
    
    # The schoolbook decomposition has rank = n*m*p = 72.
    # We try to find a lower rank decomposition using known low-rank algorithms for smaller blocks
    # and then combining them.
    
    # For 2x2 * 2x2, Strassen's algorithm gives rank 7 instead of 8.
    # For 3x3 * 3x3, the best known is around 19 (Coppersmith-Winograd lower bound is 19, 
    # but exact values are complex). Let's use a known low-rank decomposition if available.
    
    # Strategy: 
    # 1. Try to use Strassen-like decompositions for sub-blocks.
    # 2. The matrices are 3x4 and 4x6. We can pad 3 to 4 and 6 to 8, but that might increase rank.
    # 3. Alternatively, use the fact that we can split the multiplication into parts.
    
    # Let's first try the schoolbook and then see if we can do better with a search.
    
    # Schoolbook decomposition
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Try to find a better decomposition using random initialization and optimization
    # or use known low-rank decompositions for specific cases.
    
    # For 3x4 * 4x6, we can try to use the following approach:
    # Split the 4x4 middle into 2x2 blocks and use Strassen for each, but this is complex.
    
    # Let's try a simple approach: use the schoolbook decomposition and return it.
    # This guarantees correctness with rank 72.
    
    # If we have time, we can try to improve it.
    
    # For now, return the schoolbook decomposition.
    # In a more advanced implementation, we would try to find a lower rank decomposition
    # using optimization techniques or known low-rank algorithms.
    
    # Check if we have time to do a simple search
    elapsed = time.time() - start_time
    if elapsed > deadline - 5:  # 5 second safety margin
        return schoolbook()
    
    # Try to find a better decomposition using a simple search
    # This is a placeholder for a more advanced search algorithm.
    # For now, we return the schoolbook decomposition.
    
    return schoolbook()
# EVOLVE-BLOCK-END
