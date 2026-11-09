"""Symbolic regression: localized Gaussian RBF + global linear trend, linear in params."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form prediction using a hybrid basis:
    - 8 localized Gaussian RBFs centered at distinct points in [-3, 3]
    - 1 linear term (x)
    - 1 constant term
    
    The model is linear in params, so BFGS converges to the global optimum
    quickly (convex least-squares subproblem).
    
    RBFs with fixed centers/widths and learnable amplitudes provide a stable,
    non-linear basis that captures local variations better than global spectral
    or polynomial expansions.
    """
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Define 8 centers evenly spaced in [-3, 3]
    centers = np.linspace(-3.0, 3.0, 8)
    # Widths: use a moderate width to allow overlap but maintain locality
    # sigma = 0.8 is a reasonable choice for domain length 6
    sigma = 0.8
    
    # Calculate Gaussian RBFs
    # G_i(x) = exp(-(x - c_i)^2 / (2 * sigma^2))
    # We compute this efficiently using broadcasting
    # x shape: (N,), centers shape: (8,)
    # diff shape: (N, 8)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    rbf_values = np.exp(-(diff**2) / (2.0 * sigma**2))
    
    # Construct the full basis matrix
    # Columns 0-7: RBFs
    # Column 8: x
    # Column 9: 1
    basis = np.column_stack([rbf_values, x, np.ones_like(x)])
    
    return basis @ p
# EVOLVE-BLOCK-END
