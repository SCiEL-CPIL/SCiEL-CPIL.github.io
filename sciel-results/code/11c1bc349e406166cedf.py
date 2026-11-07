"""
Combined approach: Hybrid non-linear basis with adaptive width.
Uses a mix of Gaussian RBFs (with BFGS-optimized width), tanh activations,
and a constant term to break symmetry and capture diverse target structures.
The basis is linear in params (weights), so BFGS converges to global optimum.
"""
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis model with 10 parameters:
    - params[0]: constant offset
    - params[1]: log-width for Gaussian RBFs (shared)
    - params[2:6]: weights for 4 Gaussian RBF centers
    - params[6:9]: weights for 3 tanh activations (centers at 0.1, 3.05, 6.0)
    - params[9]: weight for a quadratic term x^2 (captures curvature)
    
    All basis functions are fixed (centers, scales), only weights and width are free.
    Linear in params -> BFGS global optimum.
    """
    # --- Gaussian RBFs ---
    # 4 centers evenly spaced in [0.1, 6]
    rbf_centers = np.linspace(0.1, 6.0, 4)
    
    # Width from param[1], using exp for positive width
    # Clamp to avoid extreme values that cause numerical issues
    log_width = params[1]
    width = np.exp(np.clip(log_width, -3.0, 3.0))  # width in [~0.05, ~20]
    
    # Build RBF matrix: shape (n_points, 4)
    x_expanded = x[:, np.newaxis]  # (n, 1)
    centers_expanded = rbf_centers[np.newaxis, :]  # (1, 4)
    
    dist_sq = (x_expanded - centers_expanded) ** 2  # (n, 4)
    rbf_values = np.exp(-dist_sq / (2.0 * width ** 2))  # (n, 4)
    
    # --- Tanh activations ---
    # 3 tanh units with centers at 0.1, 3.05, 6.0 and fixed scale 1.5
    tanh_centers = np.array([0.1, 3.05, 6.0])
    tanh_scale = 1.5
    tanh_values = np.tanh((x_expanded - tanh_centers[np.newaxis, :]) / tanh_scale)  # (n, 3)
    
    # --- Quadratic term ---
    quad_values = x ** 2  # (n,)
    
    # Combine all basis functions with their respective weights
    # params[0]: constant
    # params[2:6]: RBF weights
    # params[6:9]: tanh weights
    # params[9]: quadratic weight
    
    prediction = (
        params[0]
        + rbf_values @ params[2:6]
        + tanh_values @ params[6:9]
        + params[9] * quad_values
    )
    
    return prediction
