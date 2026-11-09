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
    or polynomial expansion.
    """
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # RBF centers spread across [-3, 3]
    centers = np.array([-3.0, -2.25, -1.5, -0.75, 0.0, 0.75, 1.5, 2.25, 3.0])
    
    # RBF widths - moderate width to capture local features without being too narrow
    # Using a uniform width that balances locality and smoothness
    widths = np.array([0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8])
    
    # Compute RBF values: phi_i(x) = exp(-((x - c_i)^2) / (2 * sigma_i^2))
    # centers has shape (9,), x has shape (n,)
    # We need to compute for each x and each center
    x_expanded = x[:, np.newaxis]  # (n, 1)
    centers_expanded = centers[np.newaxis, :]  # (1, 9)
    widths_expanded = widths[np.newaxis, :]  # (1, 9)
    
    # Compute squared distances
    diff = x_expanded - centers_expanded  # (n, 9)
    rbf_values = np.exp(-0.5 * (diff / widths_expanded) ** 2)  # (n, 9)
    
    # Linear and constant terms
    linear_term = x
    constant_term = np.ones_like(x)
    
    # 11 basis functions total: 9 RBFs + 1 linear + 1 constant
    # But we only have 10 params, so let's use 8 RBFs + 1 linear + 1 constant
    # Remove the last RBF center to have exactly 10 basis functions
    centers = centers[:-1]  # 8 centers
    widths = widths[:-1]  # 8 widths
    
    x_expanded = x[:, np.newaxis]
    centers_expanded = centers[np.newaxis, :]
    widths_expanded = widths[np.newaxis, :]
    
    diff = x_expanded - centers_expanded
    rbf_values = np.exp(-0.5 * (diff / widths_expanded) ** 2)
    
    # Build basis matrix: (n, 10)
    basis = np.column_stack([rbf_values, linear_term, constant_term])
    
    return basis @ p
# EVOLVE-BLOCK-END
