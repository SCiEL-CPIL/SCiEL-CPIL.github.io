"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Linear combination of 10 Gaussian Radial Basis Functions.
    
    Centers are fixed at evenly spaced points in [-2, 2].
    The model is linear in params, so BFGS (or any linear solver) will find
    the global minimum for the training data. This provides a smooth, flexible
    approximation without the oscillations of high-degree polynomials.
    """
    # Define fixed centers for the RBFs
    # We use 10 centers evenly spaced from -2 to 2
    centers = np.linspace(-2.0, 2.0, 10)
    
    # Calculate distances from each input point to each center
    # x shape: (N, 1) or (N,)
    # We need to ensure x is 1D for broadcasting ease, or handle shapes carefully.
    x_flat = np.asarray(x).ravel()
    
    # Compute squared distances: (N, 1) - (1, 10) -> (N, 10)
    # centers shape: (10,) -> (1, 10)
    dist_sq = (x_flat[:, np.newaxis] - centers[np.newaxis, :]) ** 2
    
    # Define the width (sigma) for the Gaussians.
    # A moderate width allows overlapping influence but distinct local features.
    # Domain size is 4. 10 centers -> spacing 0.44.
    # sigma ~ 0.3-0.5 is reasonable. Let's use 0.4.
    sigma = 0.4
    width_sq = 2.0 * sigma ** 2
    
    # Compute Gaussian basis values
    # basis shape: (N, 10)
    basis = np.exp(-dist_sq / width_sq)
    
    # Linear combination: (N, 10) @ (10,) -> (N,)
    prediction = basis @ params
    
    # Return shape matching input if possible, but evaluator expects array with one prediction per point.
    # If input was 2D, we should probably return 1D or same shape?
    # "equation must return an array with one prediction per point."
    # Usually 1D array of length N is safe.
    return prediction
# EVOLVE-BLOCK-END
