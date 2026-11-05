"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def _deadline_guard(start_time, limit=145.0):
    """Return True if we still have time budget remaining."""
    return (time.time() - start_time) < limit

def construct():
    """
    Returns the schoolbook tensor decomposition for 4x5 * 5x5 matrix multiplication.
    Rank = 4*5*5 = 100.
    
    Interface:
    U: shape (n*m, rank) -> (20, 100)
    V: shape (m*p, rank) -> (25, 100)
    W: shape (p*n, rank) -> (20, 100)
    
    Convention:
    U[i*m + j, r]
    V[j*p + k, r]
    W[k*n + i, r]
    """
    start_time = time.time()
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Build using numpy for structural clarity
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * m * p + j * p + k
                U[i * m + j, r] = 1.0
                V[j * p + k, r] = 1.0
                W[k * n + i, r] = 1.0
    
    # Convert to list-of-lists for the required return interface
    U_out = U.tolist()
    V_out = V.tolist()
    W_out = W.tolist()
    
    if not _deadline_guard(start_time):
        pass  # still within budget; no extra work needed
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
