"""Symbolic regression with 1 input(s)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers the target function using a hybrid basis expansion.
    
    The model combines a low-degree polynomial part (for global trends) with
    Gaussian Radial Basis Functions (for local features). 
    The 10 parameters are split:
    - params[0:4] : coefficients for polynomial basis [1, x, x^2, x^3]
    - params[4:10]: weights for 6 Gaussian RBFs
    
    This hybrid approach aims to capture both the global structure and
    potential localized features of the unknown function.
    """
    # Deadline guard: The evaluator kills after 150s.
    start_time = time.time()
    
    # Ensure x is a 1D array for broadcasting
    x = np.asarray(x, dtype=np.float64).flatten()
    p = np.asarray(params, dtype=np.float64)
    
    # Safety: ensure params has at least 10 elements
    if p.shape[0] < 10:
        p = np.pad(p, (0, 10 - p.shape[0]), constant_values=0.0)
    else:
        p = p[:10]
    
    # --- Part 1: Polynomial Basis (Global) ---
    # Polynomial terms: 1, x, x^2, x^3
    # We use the first 4 parameters for this
    poly_result = (
        p[0] * np.ones_like(x) +
        p[1] * x +
        p[2] * x**2 +
        p[3] * x**3
    )
    
    # --- Part 2: Gaussian RBF Basis (Local) ---
    # 6 RBF centers evenly distributed in [-3, 3]
    # Using the full domain to ensure edge coverage
    centers = np.linspace(-3.0, 3.0, 6)
    
    # Width of the Gaussians.
    # A width of 1.2 provides better overlap for 6 centers over domain length 6,
    # reducing the risk of underfitting in gaps between centers while maintaining
    # locality.
    sigma = 1.2
    
    # Compute the RBF basis matrix
    # shape: (num_points, 6)
    # B[i, j] = exp(-(x_i - centers[j])^2 / (2 * sigma^2))
    
    # Calculate squared distances
    # x[:, None] has shape (N, 1)
    # centers[None, :] has shape (1, 6)
    diff = x[:, None] - centers[None, :]
    
    # Compute the exponential term
    basis = np.exp(-diff**2 / (2 * sigma**2))
    
    # The model is a linear combination of the RBF basis functions
    # result_rbf = basis @ params[4:10]
    # params[4:10] has shape (6,)
    # basis has shape (N, 6)
    # result_rbf has shape (N,)
    
    rbf_result = basis @ p[4:10]
    
    # Combine
    result = poly_result + rbf_result
    
    # Safety check: ensure finite output
    if not np.all(np.isfinite(result)):
        result = np.nan_to_num(result, nan=0.0, posinf=1e10, neginf=-1e10)
    
    return result
# EVOLVE-BLOCK-END
