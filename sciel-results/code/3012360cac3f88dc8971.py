"""
Strategy: Localized Gaussian Radial Basis Functions (RBF) with global linear trend.

This strategy uses a set of localized Gaussian RBFs centered at distinct points
within the input domain, combined with a global linear trend. This differs
structurally from the current global spectral/polynomial approach by using
local interpolation rather than global expansion.

Rationale:
1. Localized basis: Each Gaussian captures features in a local neighborhood,
   allowing the model to represent sharp transitions, discontinuities, and
   non-smooth features that global bases cannot.
2. Universal approximation: RBF networks are universal approximators for
   continuous functions.
3. Linear in parameters: The model is linear in the weights, so BFGS can
   find the global optimum efficiently.
4. 10 parameters: We allocate 8 parameters to 8 Gaussian RBFs centered at
   equally spaced points in [-3, 3], and 2 parameters to a linear trend
   (intercept and slope).

Basis functions:
- 1 (intercept)
- x (slope)
- exp(-(x - c_i)^2 / (2*sigma^2)) for i = 0..7, where c_i are 8 centers
  evenly spaced in [-3, 3], and sigma is a fixed width.

The centers are at: -3, -2.14, -1.29, -0.43, 0.43, 1.29, 2.14, 3
The width sigma is chosen to be about 0.5 to provide good local coverage.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Precompute centers and sigma for the 8 Gaussian RBFs
_CENTERS = np.array([-3.0, -2.142857, -1.285714, -0.428571, 0.428571, 1.285714, 2.142857, 3.0])
_SIGMA = 0.5
_SIGMA_SQ = _SIGMA ** 2

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Computes the RBF + linear trend prediction.
    
    Args:
        x: Input array of shape (N,)
        params: Coefficients array of shape (10,)
            params[0] = intercept
            params[1] = slope
            params[2:10] = weights for 8 Gaussian RBFs
    
    Returns:
        Prediction array of shape (N,)
    """
    x = np.asarray(x).ravel()
    N = len(x)
    
    # Initialize prediction with linear trend
    pred = params[0] + params[1] * x
    
    # Add Gaussian RBF contributions
    # x has shape (N,), _CENTERS has shape (8,)
    # Compute squared distances: (N, 8)
    diff = x[:, np.newaxis] - _CENTERS[np.newaxis, :]  # shape (N, 8)
    sq_dist = diff ** 2  # shape (N, 8)
    
    # Compute Gaussian values: exp(-sq_dist / (2 * sigma^2))
    gaussians = np.exp(-sq_dist / (2.0 * _SIGMA_SQ))  # shape (N, 8)
    
    # Weighted sum of Gaussians
    pred += gaussians @ params[2:10]
    
    return pred

# EVOLVE-BLOCK-END
