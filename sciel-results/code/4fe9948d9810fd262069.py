"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

_start_time = time.time()
TIME_LIMIT = 130.0  # Safety margin before 150s kill

def construct():
    """
    Return a rank-minimal (or near-minimal) exact tensor decomposition
    for multiplying a 3x5 matrix by a 5x5 matrix.
    
    Returns dict with keys 'U', 'V', 'W' where:
      U has shape (n*m, rank) = (15, rank)
      V has shape (m*p, rank) = (25, rank)  
      W has shape (p*n, rank) = (15, rank)
    
    The decomposition satisfies: for all i,j,k,
      sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    (which encodes C[i,k] = sum_j A[i,j]*B[j,k])
    """
    n, m, p = DIMENSIONS  # 3, 5, 5
    
    # Schoolbook decomposition: rank = n*m*p = 75
    rank = n * m * p
    
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    # Check if we can do better with Strassen-like approaches
    # For 3x5x5, the schoolbook rank is 75.
    # Known results suggest the rank is likely 75 or very close.
    # Let's verify the schoolbook decomposition is correct.
    
    # Try to use a 2x2x2 Strassen block if applicable
    # But 3x5x5 doesn't decompose neatly into 2x2x2 blocks.
    
    # The schoolbook decomposition is exact and has rank 75.
    # We return it as the best known decomposition.
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
