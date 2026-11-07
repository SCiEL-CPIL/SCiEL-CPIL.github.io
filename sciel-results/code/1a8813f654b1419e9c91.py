"""
Structural change: Gaussian Radial Basis Functions (RBFs).
Instead of global periodic (Fourier) or global polynomial (Chebyshev) bases,
we use local basis functions centered evenly across the domain.
This is robust for non-periodic smooth functions and avoids the numerical
instabilities observed with the previous Chebyshev attempt.
The model is linear in the parameters, ensuring BFGS convergence is stable.
"""
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Gaussian RBF basis on [0.1, 6] with 10 parameters.
    
    1. Map x to [0, 1] for stable kernel scaling.
    2. Define 10 fixed centers evenly spaced in [0, 1].
    3. Compute Gaussian kernels with fixed width.
    4. Return linear combination of kernels weighted by params.
    """
    # Map x from [0.1, 6.0] to [0, 1]
    x_min = 0.1
    x_max = 6.0
    x_norm = (x - x_min) / (x_max - x_min)
    
    # 10 centers evenly spaced in [0, 1]
    # Using linspace(0, 1, 10)
    centers = np.linspace(0.0, 1.0, 10)
    
    # Fixed width (sigma) for the Gaussian kernels.
    # In normalized space, a sigma of ~0.15 allows good overlap between
    # adjacent centers (spacing is 1/9 ~ 0.11).
    sigma = 0.15
    
    # Compute basis matrix: (n_points, 10)
    # diff is (n_points, 10)
    diff = x_norm[:, np.newaxis] - centers[np.newaxis, :]
    basis = np.exp(-(diff ** 2) / (2.0 * sigma ** 2))
    
    return basis @ params
