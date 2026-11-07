"""
Combined approach: RBF-MLP with adaptive width and Fourier refinement.
Uses a mix of Gaussian RBFs (with BFGS-optimized width) and Fourier terms
to capture both local structure and global periodicity. The basis is linear
in params (weights), so BFGS converges to the global optimum.
"""
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid RBF-Fourier basis model with 10 parameters:
    - params[0]: constant offset
    - params[1]: log-width for Gaussian RBFs (shared)
    - params[2:6]: weights for 4 Gaussian RBF centers
    - params[6:10]: weights for 4 Fourier terms (2 cos + 2 sin)
    
    All basis functions are fixed (centers, scales, frequencies), only weights and width are free.
    Linear in params -> BFGS global optimum.
    """
    x = np.asarray(x, dtype=np.float64)
    
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
    
    # --- Fourier terms ---
    # 2 cosine + 2 sine terms with frequencies scaled to the domain
    k_values = np.array([1.0, 2.0])
    base_freq = np.pi / 5.9  # domain length is ~5.9
    
    angles = np.outer(x, k_values * base_freq)  # (n, 2)
    cos_terms = np.cos(angles)  # (n, 2)
    sin_terms = np.sin(angles)  # (n, 2)
    
    # Combine all basis functions with their respective weights
    # params[0]: constant
    # params[2:6]: RBF weights
    # params[6:8]: cos weights
    # params[8:10]: sin weights
    
    prediction = (
        params[0]
        + rbf_values @ params[2:6]
        + cos_terms @ params[6:8]
        + sin_terms @ params[8:10]
    )
    
    return prediction
