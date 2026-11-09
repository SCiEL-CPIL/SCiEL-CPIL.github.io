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
        
    # Map x from [-3, 3] to t in [-1, 1] for better numerical conditioning
    t = x / 3.0
    
    # Strategy: Radial Basis Function (RBF) model with 5 fixed centers.
    # Centers are evenly spaced in [-1, 1]: -1, -0.5, 0, 0.5, 1.
    # Params 0-4: Weights for each Gaussian.
    # Params 5-9: Log-widths (log(sigma)) for each Gaussian.
    # This is a non-linear model, but BFGS can optimize the parameters directly.
    # The non-linearity allows capturing complex shapes better than a linear basis.
    
    centers = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    
    # Compute distances from each point to each center
    # dist shape: (n_points, 5)
    dist = np.abs(t[:, np.newaxis] - centers[np.newaxis, :])
    
    # Extract weights and log-widths
    weights = params[0:5]
    log_sigmas = params[5:10]
    
    # Compute sigmas with a safety floor to prevent division by zero or overflow
    # If log_sigma is very large, sigma is large -> flat Gaussian
    # If log_sigma is very small, sigma is small -> spike
    # Clip log_sigmas to reasonable range to maintain numerical stability
    log_sigmas = np.clip(log_sigmas, -5.0, 5.0)
    sigmas = np.exp(log_sigmas)
    
    # Compute Gaussian values
    # exp(-dist^2 / (2 * sigma^2))
    # dist^2 shape: (n_points, 5)
    # sigmas shape: (5,)
    gaussians = np.exp(-dist**2 / (2.0 * sigmas**2))
    
    # Weighted sum: (n_points, 5) dot (5,) -> (n_points,)
    prediction = gaussians @ weights
    
    # Deadline check
    if time.time() - t_start > 100:
        # Fallback to linear if it takes too long (unlikely for 5 RBFs)
        return params[0] * t + params[1]
        
    return prediction
# EVOLVE-BLOCK-END
