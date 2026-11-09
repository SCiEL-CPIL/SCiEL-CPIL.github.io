"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = None
_TIME_LIMIT = 115.0  # Safety margin before 120s kill

def _check_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + _TIME_LIMIT
    return time.time() < _DEADLINE


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximate the target function using an RBF (Radial Basis Function) network
    with fixed centers and a learnable width parameter.
    
    Args:
        x: Input array of shape (N,).
        params: Array of 10 parameters.
        
    Returns:
        Prediction array of shape (N,).
    """
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Check deadline
    if not _check_deadline():
        # Return a simple fallback if we're running out of time
        return np.ones_like(x) * params[0]
    
    # Define fixed RBF centers spanning [0.1, 6]
    n_centers = 9  # Use 9 centers so we have 9 weights + 1 width = 10 params
    centers = np.linspace(0.1, 6.0, n_centers)
    
    # Extract the width parameter (sigma) from params[0]
    # Use softplus to ensure positive width, with a reasonable range
    sigma = np.exp(params[0])  # Exponential ensures positive
    sigma = np.clip(sigma, 0.05, 10.0)  # Clip to reasonable range
    
    # Extract the RBF weights
    weights = params[1:n_centers+1]
    
    # Compute RBF basis: phi(x) = exp(-((x - c)^2) / (2 * sigma^2))
    # x shape: (N,), centers shape: (n_centers,)
    # We want basis shape: (N, n_centers)
    
    # Compute squared distances
    diff = x[:, np.newaxis] - centers[np.newaxis, :]  # (N, n_centers)
    sq_dist = diff * diff  # (N, n_centers)
    
    # Compute Gaussian RBF values
    basis = np.exp(-sq_dist / (2.0 * sigma * sigma))
    
    # Check for non-finite values
    if not np.all(np.isfinite(basis)):
        basis = np.nan_to_num(basis, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Prediction is the linear combination of RBF values
    result = basis @ weights
    
    # Ensure output is finite
    if not np.all(np.isfinite(result)):
        result = np.nan_to_num(result, nan=0.0, posinf=1e10, neginf=-1e10)
    
    return result
# EVOLVE-BLOCK-END
