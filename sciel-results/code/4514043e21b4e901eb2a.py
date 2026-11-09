import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a hybrid basis:
    - 3 global monomials (1, x, x^2) for the overall trend.
    - 7 local Gaussian RBFs for local details.
    
    This hybrid approach is structurally different from pure local splines or pure global polynomials.
    The global terms capture the low-frequency components (which are likely smooth), while the local RBFs
    capture high-frequency variations or sharp features. This decomposition often leads to better conditioning
    and faster convergence for BFGS because the parameters for global and local features have distinct scales
    and sensitivities.
    
    The centers of the RBFs are fixed to be evenly spaced in the interior of the domain.
    The widths are fixed to a reasonable value to avoid overfitting to noise while allowing flexibility.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    p = params[:10]
    
    # Ensure finite output
    if not np.all(np.isfinite(p)):
        return np.zeros_like(x)
    
    # Global basis: 1, x, x^2
    # These are scaled to be O(1) to help conditioning
    # x is in [-3, 3], so x^2 is in [0, 9]. 
    # We can normalize x to [-1, 1] for the basis to improve conditioning.
    x_norm = x / 3.0
    
    T_global = np.column_stack((
        np.ones(n),
        x_norm,
        x_norm ** 2
    ))
    
    # Local basis: 7 Gaussian RBFs
    # Centers evenly spaced in (-2.5, 2.5) to avoid edge effects
    centers = np.linspace(-2.5, 2.5, 7)
    width = 0.5  # Fixed width
    
    T_local = np.zeros((n, 7))
    for i, c in enumerate(centers):
        T_local[:, i] = np.exp(-((x - c) ** 2) / (2 * width ** 2))
    
    # Combine basis
    T = np.hstack((T_global, T_local))
    
    # Prediction
    y_pred = T @ p
    
    return y_pred
