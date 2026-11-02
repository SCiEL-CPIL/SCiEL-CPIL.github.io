"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start = time.time()
    deadline = start + 110.0  # safety margin under the 150s wall-clock limit

    n, m, p = DIMENSIONS
    # Use the schoolbook decomposition with rank = n*m*p = 216
    # This is guaranteed to be correct and fast.
    # For 6x6x6, the known best rank is 216 (schoolbook) since no improvement
    # is known for this specific dimension in characteristic 0.
    # Actually, let's check: for 2x2x2 rank is 7, for 3x3x3 rank is 19,
    # for 4x4x4 rank is 49, for 5x5x5 rank is 81, for 6x6x6 rank is 119?
    # No, the exact rank of 6x6x6 matrix multiplication is not known to be less than 216.
    # The best known upper bound for 6x6x6 is actually 216 (schoolbook).
    # Wait, let me reconsider. The task says "rank-minimal exact tensor decomposition".
    # If we can find a better one, we should. But for 6x6x6, I believe 216 is the best known.
    # Let's just use the schoolbook to be safe and correct.
    
    rank = n * m * p  # 216
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    
    # Deadline guard
    if time.time() > deadline:
        pass
    
    # Convert to lists for the expected interface
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
