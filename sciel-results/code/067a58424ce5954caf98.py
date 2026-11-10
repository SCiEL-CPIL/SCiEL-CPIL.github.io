import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a Gaussian Radial Basis Function (RBF) model.
    The model is linear in the 10 parameters, ensuring BFGS converges to the global minimum.
    """
    # Safety margin for the 150s limit
    start_time = time.time()
    deadline = 140.0  # 10s buffer
    
    # 10 centers evenly spaced in [-2, 2]
    centers = np.linspace(-2.0, 2.0, 10)
    
    # Width parameter
    # Tuned to balance locality and smoothness
    sigma = 0.35
    sigma_sq_2 = 2.0 * sigma * sigma
    
    # Check deadline (unlikely to trigger, but required by directive)
    if time.time() > deadline:
        # Fallback to linear model if computation is unexpectedly slow
        return params[0] * x + params[1]
        
    # Compute Gaussian basis: exp(-(x - c)^2 / (2 * sigma^2))
    # x: (N,), centers: (10,) -> diff: (N, 10)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    basis = np.exp(-diff * diff / sigma_sq_2)
    
    # Linear combination of basis functions
    pred = basis @ params
    
    return pred
