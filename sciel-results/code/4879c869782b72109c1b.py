# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression: recover an unknown function of 1 input.
    
    Uses a linear combination of 10 RBFs at fixed, evenly spaced centers.
    The first parameter controls the shared width (sigma) of all kernels.
    The remaining 9 parameters are coefficients for the 9 RBFs.
    
    This design is smooth, well-conditioned, and linear in the coefficient
    parameters (though nonlinear in sigma). BFGS can optimize this efficiently.
    Starting from all-ones, sigma=1.0 is a reasonable initial width.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Guard against non-finite inputs
    x = np.where(np.isfinite(x), x, 0.0)
    
    # Extract width parameter (first param) and coefficients (next 9)
    sigma = params[0]
    coeffs = params[1:]
    
    # Clamp sigma to a reasonable range to avoid numerical issues
    # sigma should be positive; if it goes negative, use absolute value
    # but also cap it to avoid extremely wide/narrow kernels
    sigma = np.abs(sigma) + 1e-6
    sigma = np.clip(sigma, 0.1, 10.0)
    
    # Define 9 evenly spaced centers in [-3, 3]
    # Centers: -3, -2, -1, 0, 1, 2, 3 (7 centers) + 2 more?
    # Let's use 9 centers evenly spaced in [-3, 3]
    centers = np.linspace(-3.0, 3.0, 9)
    
    # Compute RBF values: exp(-((x - c)^2) / (2 * sigma^2))
    # x shape: (N,), centers shape: (9,)
    # diff shape: (N, 9)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    
    # Compute squared distances
    sq_dist = diff ** 2
    
    # Compute RBF values
    rbf_vals = np.exp(-sq_dist / (2.0 * sigma ** 2))
    
    # Linear combination: rbf_vals @ coeffs
    result = rbf_vals @ coeffs
    
    # Ensure finite output
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
# EVOLVE-BLOCK-END
