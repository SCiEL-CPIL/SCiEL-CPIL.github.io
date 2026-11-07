"""Program using a Kernel Ridge Regression strategy with Gaussian RBF kernels."""
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form symbolic regression candidate.
    Uses a linear combination of 10 Gaussian RBF kernels with fixed centers.
    The 10 params act as adaptive weights for local interpolation.
    
    This shifts from global functional basis (polynomials/trig) to local flexibility,
    enabling the model to capture arbitrary smooth functions by weighting local bumps.
    """
    # Scale input to [-1, 1] for better kernel behavior
    u = x / 3.0

    # Define 10 fixed, evenly spaced centers in [-1, 1]
    # Using linspace from -0.9 to 0.9 to avoid edge effects
    centers = np.linspace(-0.9, 0.9, 10)

    # Define a fixed bandwidth (sigma) for the Gaussian kernels.
    # A moderate bandwidth allows overlap for smooth interpolation.
    sigma = 0.35
    sigma2 = sigma * sigma

    # Compute the RBF basis: exp(-0.5 * (u - c)^2 / sigma^2)
    # u has shape (N,), centers has shape (10,)
    # We want a matrix of shape (N, 10)
    # (u[:, np.newaxis] - centers[np.newaxis, :])
    diff = u[:, np.newaxis] - centers[np.newaxis, :]
    
    # Gaussian kernel values
    rbf = np.exp(-0.5 * (diff ** 2) / sigma2)

    # Linear combination with params
    # rbf shape: (N, 10), params shape: (10,)
    y = rbf @ params

    return y
