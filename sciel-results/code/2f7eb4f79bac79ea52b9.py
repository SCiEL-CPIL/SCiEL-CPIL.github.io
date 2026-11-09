"""Symbolic regression with linear basis of diverse normalized functions."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Linear parametric model with 10 diverse basis functions.
    
    The model is linear in parameters, which makes the optimization problem
    convex (least squares). BFGS will reliably find the global optimum.
    
    Basis functions are chosen to span a wide space of common function types:
    - Polynomials (degrees 0-3, normalized)
    - Gaussian bump (centered)
    - tanh (smooth step)
    - Sine (oscillation)
    - Lorentzian-like (even and odd)
    - Exponential decay
    
    All basis functions are O(1) on [-3, 3] for numerical stability.
    
    params: [c0, c1, ..., c9] - coefficients for each basis function
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Clip x for safety
    x_safe = np.clip(x, -10, 10)
    
    # Basis 0: constant
    b0 = np.ones_like(x_safe)
    
    # Basis 1: linear, normalized
    b1 = x_safe / 3.0
    
    # Basis 2: quadratic, normalized
    b2 = (x_safe ** 2) / 9.0
    
    # Basis 3: cubic, normalized
    b3 = (x_safe ** 3) / 27.0
    
    # Basis 4: Gaussian bump (centered, width ~1)
    b4 = np.exp(-(x_safe ** 2))
    
    # Basis 5: tanh (smooth odd function)
    b5 = np.tanh(x_safe)
    
    # Basis 6: sine
    b6 = np.sin(x_safe)
    
    # Basis 7: Lorentzian-like (even)
    b7 = 1.0 / (1.0 + x_safe ** 2)
    
    # Basis 8: odd Lorentzian-like
    b8 = x_safe / (1.0 + x_safe ** 2)
    
    # Basis 9: exponential decay (even)
    b9 = np.exp(-np.abs(x_safe))
    
    # Combine with parameters (linear combination)
    pred = (params[0] * b0 + params[1] * b1 + params[2] * b2 +
            params[3] * b3 + params[4] * b4 + params[5] * b5 +
            params[6] * b6 + params[7] * b7 + params[8] * b8 +
            params[9] * b9)
    
    # Ensure finite output
    pred = np.nan_to_num(pred, nan=0.0, posinf=1e10, neginf=-1e10)
    
    return pred
# EVOLVE-BLOCK-END
