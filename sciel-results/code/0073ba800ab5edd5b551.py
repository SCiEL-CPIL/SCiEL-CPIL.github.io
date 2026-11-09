import time
import numpy as np

DIMENSIONS = [3, 5, 7]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3 by 5 matrix by a 5 by 7 matrix.
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices.
    The score is the rank (lower is better).
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Initialize with schoolbook decomposition
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    
    U = np.zeros((n * m, schoolbook_rank), dtype=np.float64)
    V = np.zeros((m * p, schoolbook_rank), dtype=np.float64)
    W = np.zeros((p * n, schoolbook_rank), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U[i*m + j, col] = 1
                # V[j*p + k, col] = 1
                # W[k*n + i, col] = 1
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
                
    # Convert to lists for JSON serialization if required by evaluator, 
    # though the interface says "factor matrices", usually numpy arrays or lists are accepted.
    # The seed program used .tolist(), so I will stick to that for compatibility.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
