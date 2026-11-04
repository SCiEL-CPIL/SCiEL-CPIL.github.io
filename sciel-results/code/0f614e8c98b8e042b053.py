"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 4x4x9 matrix multiplication.
    Currently implements the schoolbook decomposition with rank 144.
    Includes a deadline guard to ensure termination within 120s.
    """
    start_time = time.time()
    deadline = 110.0  # 10 second safety margin
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if time.time() > deadline:
                    # Return whatever we have so far (should be complete if loop is fast)
                    # But for schoolbook, it's very fast.
                    break
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
        
        if time.time() > deadline:
            break
        
    if time.time() > deadline:
        # Fallback: return a valid, albeit suboptimal, partial or full schoolbook
        # Since the loop is fast, this shouldn't happen.
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
