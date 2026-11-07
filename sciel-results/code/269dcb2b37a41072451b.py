"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Uses a Parametric Radial Basis Function (RBF) network with 5 Gaussian centers.
    The 10 parameters are split into:
    - p0..p4: Raw center locations, mapped via sigmoid to [0.1, 6.0]
    - p5..p9: Amplitudes of the 5 Gaussian basis functions
    
    This allows the optimizer to place the 'bumps' where they are most useful 
    for approximating the target function, rather than being constrained to 
    fixed evenly-spaced locations. Fixed widths are used to maintain stability.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Split parameters
    center_params = params[:5]
    amp_params = params[5:]
    
    # Map raw center params to [0.1, 6.0] using sigmoid
    # sigmoid(z) = 1 / (1 + exp(-z))
    # We want to avoid saturation where gradients vanish, so we don't scale z too much
    # But sigmoid(0) = 0.5, which maps to center 3.05.
    # To ensure good coverage, we might want to encourage spread, but BFGS will handle that.
    centers = 0.1 + 5.9 * (1.0 / (1.0 + np.exp(-center_params)))
    
    # Fixed width for all Gaussians
    # A width of 1.0 covers about 2 units (1 sigma). With 5 centers in a 5.9 range, 
    # they will overlap significantly, providing a smooth approximation.
    sigma = 1.0
    
    # Compute the Gaussian basis functions
    # x is shape (N,), centers is shape (5,)
    # diff is shape (N, 5)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    
    # Compute exp(-0.5 * (diff/sigma)^2)
    basis = np.exp(-0.5 * (diff / sigma) ** 2)
    
    # Linear combination: y = basis @ amp_params
    # basis is (N, 5), amp_params is (5,)
    y = basis @ amp_params
    
    return y
# EVOLVE-BLOCK-END
