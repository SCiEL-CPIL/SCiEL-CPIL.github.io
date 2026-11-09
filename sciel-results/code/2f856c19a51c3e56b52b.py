import numpy as np
import time
import sys
from scipy.linalg import eigh

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 600.0  # 600s limit, keep margin
    
    # Build the Jacobi matrix for Gaussian quadrature with weight x^12 * e^{-x}
    # N = 12 points, alpha = 12
    N = 12
    alpha = 12
    
    J = np.zeros((N, N), dtype=float)
    
    # Diagonal entries
    for i in range(N):
        J[i, i] = 2 * i + 1 + alpha
    
    # Off-diagonal entries
    for i in range(N - 1):
        J[i, i + 1] = np.sqrt((i + 1) * (i + 1 + alpha))
        J[i + 1, i] = J[i, i + 1]
    
    # Compute eigenvalues (these are the Gaussian quadrature nodes)
    try:
        eigenvalues = eigh(J, eigvals_only=True)
        # Ensure positive and sort
        eigenvalues = np.sort(np.maximum(eigenvalues, 1e-10))
        
        # Check deadline
        if time.time() > deadline - 10.0:
            # Fallback to seed
            return [4.0 * (i + 1) for i in range(12)]
        
        return list(eigenvalues)
        
    except Exception as e:
        # Fallback to seed if eigenvalue computation fails
        return [4.0 * (i + 1) for i in range(12)]
# EVOLVE-BLOCK-END
