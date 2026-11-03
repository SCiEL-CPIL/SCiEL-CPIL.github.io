"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """Find a rank-minimal exact tensor decomposition for 2x3 by 3x4 matrix multiplication."""
    n, m, p = DIMENSIONS  # n=2, m=3, p=4
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # Start with schoolbook decomposition
    rank_schoolbook = n * m * p  # 24
    U = np.zeros((n * m, rank_schoolbook))
    V = np.zeros((m * p, rank_schoolbook))
    W = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Try to reduce rank using Winograd-style frequency domain decomposition
    # Use 6-point DFT-based approach
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try different transform sizes
    for transform_size in [6, 8, 12]:
        if time.time() > deadline:
            break
            
        # Create DFT matrix
        omega = np.exp(-2j * np.pi / transform_size)
        F = np.array([[omega**(i * j) for j in range(transform_size)] for i in range(transform_size)])
        
        # Try to use this transform to reduce rank
        # The idea: transform A along rows, B along columns, do pointwise mult, inverse transform
        # This gives a decomposition where rank = transform_size * (number of independent freq bins)
        
        # For now, just try to find linear dependencies in the schoolbook decomposition
        # using SVD on the tensor
        pass
    
    # Convert to integer/half-integer if possible
    def snap_to_half_integer(arr):
        """Snap values to nearest half-integer."""
        return np.round(arr * 2) / 2
    
    # Return the best decomposition found
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(), 
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
