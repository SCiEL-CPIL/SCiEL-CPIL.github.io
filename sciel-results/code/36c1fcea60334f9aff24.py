"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Deadline guard: stop 10 seconds before the 150s limit
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Try to find a decomposition with rank < schoolbook_rank
    # Using a simple search: try to remove one term and check if the tensor is still rank-deficient
    # This is a heuristic; for small ranks we can try linear algebra approaches
    
    # For now, return the schoolbook decomposition as a baseline
    # A more sophisticated search would attempt to reduce the rank
    
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
