"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Radial Basis Function (Gaussian) model with 10 fixed centers.
    
    This uses a local basis function approach, which can better capture
    non-global features (like bumps or wiggles) compared to a global
    polynomial basis. The centers are uniformly distributed across the
    input domain [-2, 2].
    
    params[0..9] are the weights for the 10 Gaussian RBFs.
    """
    start_time = time.time()
    
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    n = len(x)
    if n == 0:
        return np.array([], dtype=np.float64)

    # Define 10 centers uniformly spaced in [-2, 2]
    # Centers: -1.8, -1.4, ..., 1.8
    centers = np.linspace(-1.8, 1.8, 10)
    
    # Fixed width (sigma) for the Gaussians
    # A width of 0.6 provides a balance between local detail and smoothness
    sigma = 0.6
    
    # Compute RBF values
    # Shape of x: (n,)
    # Shape of centers: (10,)
    # We want (n, 10) matrix of RBF values
    
    # Efficient computation:
    # diff = x[:, np.newaxis] - centers[np.newaxis, :]
    # rbf_values = np.exp(-diff**2 / (2 * sigma**2))
    
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    rbf_values = np.exp(-diff**2 / (2.0 * sigma**2))
    
    # Compute result as linear combination
    # result = rbf_values @ params
    result = rbf_values.dot(params)
    
    # Ensure finite output
    if not np.all(np.isfinite(result)):
        result = np.where(np.isfinite(result), result, 0.0)
        
    # Deadline guard
    elapsed = time.time() - start_time
    if elapsed > 140:
        pass # Return current result
        
    return result
# EVOLVE-BLOCK-END
