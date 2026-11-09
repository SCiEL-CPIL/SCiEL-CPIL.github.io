"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid RBF + polynomial + trigonometric function approximator.
    
    Strategy: Combine three complementary basis families:
    1. Gaussian RBFs with 3 fixed centers (3 weight params)
    2. Low-degree polynomial (up to degree 3) (4 params)
    3. Trigonometric terms sin(x) and cos(x) (2 params)
    4. Constant offset (1 param)
    
    Total: 10 parameters
    
    The RBFs capture localized features, polynomials capture global trends,
    and trig terms capture periodic components. This hybrid representation
    is more flexible than any single family while remaining well-conditioned
    for BFGS optimization.
    """
    p = params.copy()
    
    # Clip parameters for numerical stability
    p = np.clip(p, -10000, 10000)
    
    # Scale x to [-1, 1] for better conditioning
    xs = x / 3.0
    
    # --- RBF component: 3 Gaussians at fixed centers ---
    # Centers in scaled space: -1, 0, 1 (corresponding to -3, 0, 3 in original)
    rbf_centers = np.array([-1.0, 0.0, 1.0])
    rbf_weights = p[0:3]
    
    # Fixed width for RBFs (moderate spread)
    rbf_width = 0.5
    x_col = xs[:, np.newaxis]
    centers_row = rbf_centers[np.newaxis, :]
    dist_sq = (x_col - centers_row) ** 2
    rbf_values = np.exp(-dist_sq / (2.0 * rbf_width ** 2))
    rbf_result = rbf_values @ rbf_weights
    
    # --- Polynomial component: up to degree 3 ---
    # Coefficients: p[3], p[4], p[5], p[6]
    poly = (p[3] + p[4] * xs + p[5] * xs**2 + p[6] * xs**3)
    
    # --- Trigonometric component ---
    # Coefficients: p[7], p[8]
    trig = p[7] * np.sin(xs * np.pi) + p[8] * np.cos(xs * np.pi)
    
    # --- Constant offset ---
    const = p[9]
    
    result = rbf_result + poly + trig + const
    
    # Ensure finite output
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
# EVOLVE-BLOCK-END
