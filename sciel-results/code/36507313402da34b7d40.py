import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers the target function using a Gaussian Radial Basis Function (RBF) network.
    
    The 10 parameters are split into:
    - params[0:5]: centers of 5 Gaussian kernels (c_1, ..., c_5)
    - params[5:10]: amplitudes of the 5 Gaussian kernels (a_1, ..., a_5)
    
    The prediction is:
        y = sum_{i=1}^{5} a_i * exp(-(x - c_i)^2 / (2 * sigma^2))
    
    where sigma is a fixed bandwidth parameter (chosen as 0.5 for good local flexibility).
    
    This local, non-parametric basis can capture sharp features, discontinuities, and
    complex non-polynomial structures that a global polynomial basis cannot.
    
    A time guard is included to ensure the function returns within the 150s wall-clock limit.
    """
    # Time guard: track elapsed time and return a safe default if approaching the limit
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin before 150s limit
    
    original_shape = x.shape
    x_flat = x.flatten()
    N = len(x_flat)
    
    if N == 0:
        return np.empty(original_shape)
    
    p = params.flatten()
    
    # Extract centers and amplitudes
    centers = p[0:5]
    amplitudes = p[5:10]
    
    # Fixed bandwidth for the Gaussian kernels
    sigma = 0.5
    
    # Compute the RBF prediction
    # For each center c_i, compute exp(-(x - c_i)^2 / (2 * sigma^2))
    y = np.zeros(N, dtype=float)
    
    for i in range(5):
        # Check time periodically
        if time.time() - start_time > DEADLINE:
            # Return partial result as a safe fallback
            return y.reshape(original_shape)
        
        diff = x_flat - centers[i]
        y += amplitudes[i] * np.exp(-0.5 * (diff / sigma) ** 2)
    
    return y.reshape(original_shape)
