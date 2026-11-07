# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Uses a Radial Basis Function (RBF) mixture with fixed centers and widths.
    This is a structural change from global polynomial/Fourier bases to a
    local basis, which can better capture sharp features or non-smooth
    behavior. The model is linear in the weights (params), ensuring a
    convex optimization landscape for BFGS.
    """
    start_time = time.time()
    timeout_limit = 115.0
    
    n_points = x.shape[0]
    
    # Define 10 fixed centers in [0.1, 6]
    # Uniformly spaced
    centers = np.linspace(0.1, 6.0, 10)
    
    # Width parameter
    sigma = 0.5
    
    # Compute RBF values
    # x shape (n_points, 1), centers shape (10, 1)
    # diff shape (n_points, 10)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    
    # exp(-diff^2 / (2 * sigma^2))
    basis = np.exp(-diff**2 / (2.0 * sigma**2))
    
    # Standardize?
    # RBFs are positive and bounded.
    # The columns might be correlated.
    # Let's not standardize initially, as the values are O(1).
    # If condition number is bad, BFGS might struggle.
    # But with 10 points and 10 centers, it might be ill-conditioned if centers are close.
    # Distance between centers is ~0.65.
    # Sigma 0.5 means significant overlap.
    # This could be ill-conditioned.
    # Let's use a smaller sigma to reduce overlap?
    # If sigma is small, the basis becomes sparse, condition number improves.
    # Let's try sigma = 0.3.
    
    # Actually, let's stick to the previous idea of fixing the Chebyshev standardization
    # and maybe trying a **Higher Degree** or **Different Polynomial** if possible?
    # No, 10 params limit the degree.
    
    # Let's try the RBF with sigma=0.4.
    sigma = 0.4
    basis = np.exp(-diff**2 / (2.0 * sigma**2))
    
    return basis @ params
# EVOLVE-BLOCK-END
