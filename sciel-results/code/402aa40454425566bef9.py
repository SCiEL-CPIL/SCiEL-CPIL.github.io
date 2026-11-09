"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Deadline guard
    t_start = time.time()
    
    n_points = len(x)
    if n_points == 0:
        return np.array([])
        
    # Strategy: Gaussian Radial Basis Functions (RBFs)
    # RBFs provide a flexible, smooth basis that is well-conditioned and
    # easier for BFGS to optimize than high-degree polynomials.
    
    # Centers evenly spaced across the input domain [-3, 3]
    centers = np.linspace(-3.0, 3.0, 10)
    
    # Width of the Gaussian kernels. 
    # sigma=1.0 provides good overlap between neighbors for smooth interpolation.
    sigma = 1.0
    
    # Compute the basis matrix: shape (10, n_points)
    # diff[i, j] = x[j] - centers[i]
    diff = x[None, :] - centers[:, None]
    
    # Gaussian kernel: exp(-0.5 * (diff/sigma)^2)
    basis = np.exp(-0.5 * (diff / sigma) ** 2)
    
    # Check time to ensure we don't exceed limits, though this is O(10*n) and very fast
    if time.time() - t_start > 100:
        # Fallback to a simple linear model if something unexpected is slow
        return params[0] * x + params[1]
        
    # Linear combination of basis functions
    return np.einsum('i,ij->j', params, basis)
# EVOLVE-BLOCK-END
