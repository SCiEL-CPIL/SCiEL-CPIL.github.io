"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
import time

def _construct_schoolbook(n, m, p):
    """
    Construct the standard schoolbook matrix multiplication decomposition.
    Returns factor matrices with integer entries (0s and 1s).
    """
    rank = n * m * p
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

def construct():
    """
    Entry point that returns a valid tensor decomposition for matrix multiplication.
    Currently returns the schoolbook decomposition with rank n*m*p = 120.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Deadline guard: ensure we finish well within the 150s limit
    deadline = start_time + 140.0
    
    # Start with the guaranteed-valid schoolbook decomposition
    result = _construct_schoolbook(n, m, p)
    
    # Check if we're still within time budget
    if time.time() < deadline:
        # Future optimization attempts would go here,
        # but must produce integer or half-integer entries
        pass
    
    return result
# EVOLVE-BLOCK-END
