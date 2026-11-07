"""Symbolic regression using RBF basis for stability and flexibility."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression: recover an unknown function of 1 input.
    
    The evaluator fits `params` (length 10) via BFGS starting from all ones,
    minimizing training MSE. We use a Radial Basis Function (RBF) model with
    9 coefficients for 9 fixed centers and 1 shared width parameter.
    
    Parameter allocation (10 total):
      params[0]: Coefficient for RBF at center c_0
      params[1]: Coefficient for RBF at center c_1
      ...
      params[8]: Coefficient for RBF at center c_8
      params[9]: Width parameter (sigma) for all RBFs
      
    Centers are evenly spaced in [-3, 3].
    RBF: phi(x, c, sigma) = exp(-0.5 * ((x - c) / sigma)^2)
    
    This design provides a smooth, universal approximator with well-conditioned
    gradients for BFGS, avoiding the instability of learnable frequencies or
    high-degree polynomials.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Extract parameters
    coeffs = params[:9]
    sigma = params[9]
    
    # Ensure sigma is positive and not too small/large
    # Clip sigma to [0.1, 10] for numerical stability
    sigma = np.clip(sigma, 0.1, 10.0)
    
    # Define 9 evenly spaced centers in [-3, 3]
    centers = np.linspace(-3.0, 3.0, 9)
    
    # Compute RBF values: shape (9, N) where N = len(x)
    # For each center, compute exp(-0.5 * ((x - c) / sigma)^2)
    # Vectorized computation:
    # diff: (9, N)
    # squared: (9, N)
    # scaled: (9, N)
    # exp: (9, N)
    
    # x has shape (N,), centers has shape (9,)
    # diff[i, j] = x[j] - centers[i]
    diff = x[np.newaxis, :] - centers[:, np.newaxis]  # (9, N)
    squared = diff ** 2  # (9, N)
    scaled = squared / (sigma ** 2)  # (9, N)
    
    # Clip the argument to prevent overflow in exp (though negative exp is safe, 
    # large positive exp is not, but here it's always <= 0 in the exponent if we 
    # write it as -0.5 * scaled)
    # Actually, exp(-0.5 * scaled) is always <= 1, so no overflow risk for large scaled.
    # But if sigma is very small, scaled can be huge, exp(-huge) -> 0, which is fine.
    # However, if sigma is extremely small, 1/sigma^2 can overflow.
    # We clipped sigma >= 0.1, so sigma^2 >= 0.01, 1/sigma^2 <= 100.
    # Max diff is 6 (from -3 to 3), so squared max is 36.
    # Max scaled is 36 * 100 = 3600.
    # exp(-0.5 * 3600) is 0. This is safe.
    
    rbf_values = np.exp(-0.5 * scaled)  # (9, N)
    
    # Combine: result = coeffs @ rbf_values
    # coeffs has shape (9,), rbf_values has shape (9, N)
    result = coeffs @ rbf_values  # (N,)
    
    # Ensure finite output
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
# EVOLVE-BLOCK-END
