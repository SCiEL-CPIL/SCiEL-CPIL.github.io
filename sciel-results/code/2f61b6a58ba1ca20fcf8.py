"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 145.0

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Adaptive Gaussian RBF Model.
    params[0:3]: Centers of the 3 Gaussian bumps
    params[3:6]: Widths (sigmas) of the 3 Gaussian bumps
    params[6:10]: Amplitudes (weights) of the 3 Gaussian bumps + 1 bias
    
    This approach allows the basis functions themselves to move and reshape
    during optimization, potentially finding a better fit for arbitrary smooth
    functions on [-3, 3] than a fixed polynomial or Fourier basis.
    """
    if time.time() - _START_TIME > _DEADLINE:
        return np.zeros_like(x, dtype=np.float64)
    
    x = np.asarray(x, dtype=np.float64)
    
    # Extract parameters
    centers = params[0:3]
    sigmas = params[3:6]
    amplitudes = params[6:9]
    bias = params[9]
    
    # Ensure sigmas are positive to avoid division by zero or negative variance
    # Using a softplus-like or just clipping. Clipping is faster and stable enough for BFGS
    # if initialized well. We'll use a small epsilon floor.
    sigmas = np.maximum(sigmas, 0.1)
    
    # Compute RBFs
    # Vectorized computation:
    # x shape: (N, 1) or (N,)
    # centers shape: (3,)
    # We want a matrix of distances: (N, 3)
    
    x_flat = x.flatten()
    
    # (N, 1) - (1, 3) -> (N, 3)
    distances = x_flat[:, np.newaxis] - centers[np.newaxis, :]
    
    # (N, 3)
    # sigma shape (3,) -> broadcast to (1, 3)
    sigmas_b = sigmas[np.newaxis, :]
    
    # Gaussian: exp(-0.5 * (d/sigma)^2)
    # (N, 3)
    rbf_values = np.exp(-0.5 * (distances / sigmas_b) ** 2)
    
    # (N, 3) * (3,) -> (N,)
    prediction = np.sum(rbf_values * amplitudes, axis=1) + bias
    
    # Reshape back to original x shape if needed, though evaluator usually expects 1D or same shape
    # The contract says "return an array with one prediction per point"
    # If x was 2D, we should probably return 2D? 
    # "equation(x, params) -> np.ndarray"
    # Usually x is 1D in these tasks, but let's be safe.
    return prediction.reshape(x.shape)
# EVOLVE-BLOCK-END
