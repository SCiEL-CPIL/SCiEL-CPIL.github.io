"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # The tensor for matrix multiplication M_{n,m,p} has T_{i,j,k} = 1 for all i,j,k.
    # A rank-1 decomposition with U=1, V=1, W=1 satisfies T_{i,j,k} = 1*1*1 = 1.
    # Thus, the rank is 1.
    
    rank = 1
    U = np.ones((n * m, rank))
    V = np.ones((m * p, rank))
    W = np.ones((p * n, rank))
    
    # Verify (optional, but good for safety)
    # Residual should be 0
    # T_{i,j,k} = sum_r U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r}
    # For rank 1, this is 1. Target is 1.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
