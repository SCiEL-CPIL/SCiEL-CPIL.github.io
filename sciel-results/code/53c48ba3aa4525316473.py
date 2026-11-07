"""
RBF Regression with Fixed Centers and Optimized Weights.
This model uses a set of Gaussian basis functions with fixed centers 
and widths. The 10 parameters are the weights (amplitudes) of these 
basis functions. This formulation is linear in the parameters, ensuring 
that BFGS will converge to the global minimum, avoiding local minima 
issues associated with optimizing centers.
"""
import numpy as np

# Precompute fixed basis parameters
# 10 centers evenly distributed in [-2, 2]
CENTERS = np.linspace(-2.0, 2.0, 10)
# Use a mix of widths to capture different scales of variation
# Wider Gaussians for global trends, narrower for local details
WIDTHS = np.array([2.0, 1.5, 1.2, 1.0, 0.8, 0.8, 1.0, 1.2, 1.5, 2.0])

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Computes the RBF prediction.
    x: Input array of shape (N,)
    params: Array of 10 weights
    """
    # x shape (N,), CENTERS shape (10,)
    # diff shape (N, 10)
    diff = x[:, np.newaxis] - CENTERS[np.newaxis, :]
    
    # Compute Gaussian basis values
    # sigma_i shape (10,) -> broadcast to (N, 10)
    sigma_sq = WIDTHS[np.newaxis, :] ** 2
    basis = np.exp(-0.5 * (diff ** 2) / sigma_sq)
    
    # Linear combination of basis functions
    # basis shape (N, 10), params shape (10,)
    return basis @ params
