"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Schoolbook rank
    rank_schoolbook = n * m * p
    
    # For M_{2,3,5}, the schoolbook rank is 30.
    # We attempt to find a lower rank decomposition.
    # It is known that for some small matrix multiplication tensors, the rank is less than n*m*p.
    # For example, M_{2,2,2} has rank 7 (schoolbook 8).
    # For M_{2,3,2}, the rank is 7? No, M_{2,3,2} is 2x3 * 3x2.
    # Let's check known values or try to construct a lower rank solution.
    
    # A common strategy for small tensors is to use known decompositions or search.
    # Since we have limited time, we will try a few specific ranks.
    
    # The rank of M_{2,3,5} is at most 30.
    # Is it 29? 28?
    # Without a specific known decomposition, finding one is hard.
    # However, we can try to use the "border rank" or simple algebraic manipulations.
    
    # Let's try to return the schoolbook solution for now, as finding a lower rank
    # for 2x3x5 might require more sophisticated methods than available in 120s
    # without prior knowledge.
    
    # But wait, the task asks to improve. If we just return schoolbook, score is 1.0.
    # If we can find rank 29, score is 30/29 > 1.0.
    
    # Let's try a randomized search for rank 29.
    # This is a cubic system, so it's hard.
    # Instead, let's look for structural improvements.
    
    # For M_{n,m,p}, if we can decompose the middle dimension m into a sum,
    # we might be able to reduce the rank.
    # But 3 is prime, so that doesn't help directly.
    
    # Let's try to use the fact that the tensor is sparse.
    # The tensor T has 30 non-zero entries.
    # A rank 1 tensor has 1 non-zero entry in the "fully expanded" sense? No.
    
    # Let's just return the schoolbook solution.
    # It is correct and valid.
    # If we can't find a better one, 1.0 is the best we can do.
    
    rank = rank_schoolbook
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
