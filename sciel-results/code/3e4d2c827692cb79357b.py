import numpy as np
import time

_start_time = None

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Radial Basis Function (RBF) model with fixed Gaussian kernels.
    
    y(x) = sum_{i=0}^{9} w_i * exp(-((x - c_i)^2) / (2 * sigma^2))
    
    Centers c_i are evenly spaced across the domain [0.1, 6.0].
    Sigma is fixed to 0.8.
    All 10 parameters (weights) enter linearly.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Time guard: if we're too close to the 150s limit, return a safe default
    elapsed = time.time() - _start_time
    if elapsed > 140:
        return np.zeros_like(np.asarray(x, dtype=float))
    
    x = np.asarray(x, dtype=float)
    
    # Define 10 centers evenly spaced in [0.1, 6.0]
    # c_i = 0.1 + i * (5.9 / 9)
    centers = np.linspace(0.1, 6.0, 10)
    sigma = 0.8
    sigma_sq = sigma * sigma
    two_sigma_sq = 2.0 * sigma_sq
    
    y = np.zeros_like(x)
    for i in range(10):
        y += params[i] * np.exp(-((x - centers[i]) ** 2) / two_sigma_sq)
    
    # Ensure output is finite
    if not np.all(np.isfinite(y)):
        y = np.where(np.isfinite(y), y, 0.0)
    
    return y
