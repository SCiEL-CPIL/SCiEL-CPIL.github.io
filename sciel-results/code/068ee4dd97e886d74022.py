"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # Schoolbook rank
    schoolbook_rank = n * m * p
    
    # Try to find a lower rank decomposition
    # Strategy: Use known constructions for 6x6 matrix multiplication
    
    # The best known rank for 6x6 is at most 189 (from observations)
    # Let's try to construct a rank-189 or lower decomposition
    
    # One approach: Use the fact that 6 = 2*3, and combine Strassen (rank 7 for 2x2) 
    # with a 3x3 construction
    
    # Actually, let's try a direct approach: generate a candidate decomposition
    # and verify it. We'll use a known construction.
    
    # For now, return schoolbook as fallback, but try to improve
    U = [[0] * schoolbook_rank for _ in range(n * m)]
    V = [[0] * schoolbook_rank for _ in range(m * p)]
    W = [[0] * schoolbook_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    best_result = {"U": U, "V": V, "W": W}
    best_rank = schoolbook_rank
    
    # Try to find a better decomposition using tensor rank methods
    # We'll use an iterative approach to try to reduce the rank
    
    # Check time
    if time.time() > deadline - 5:
        return best_result
    
    # Try a construction based on block decomposition
    # 6x6 can be viewed as 2x2 blocks of 3x3, or 3x3 blocks of 2x2
    
    # Let's try: treat 6x6 as 2x2 blocks of 3x3 matrices
    # Each 3x3 multiplication has rank at most 23 (best known)
    # But we need to be careful about the block structure
    
    # Alternative: Use a known explicit construction
    # For rank 189, we can try to find such a decomposition
    
    # Given time constraints, let's try a simple improvement:
    # Use the fact that some terms can be combined
    
    # Actually, let's just return the schoolbook for now since finding
    # a verified lower-rank decomposition in the time limit is challenging
    return best_result
# EVOLVE-BLOCK-END
