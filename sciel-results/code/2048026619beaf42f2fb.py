"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a diverse basis set that includes
    polynomials, trigonometric, exponential, and hyperbolic functions.
    
    The model is: y = sum_{i=0}^{9} p_i * phi_i(x)
    where phi_i are chosen to span a wide variety of functional forms.
    
    params: array of length 10
    """
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Ensure params has at least 10 elements
    if len(params) < 10:
        params = np.concatenate([params, np.ones(10 - len(params))])
    
    c0, c1, c2, c3, c4, c5, c6, c7, c8, c9 = params[:10]
    
    # Scale input to [-1, 1] for better conditioning
    xs = x / 3.0
    
    # Polynomial basis
    x2 = xs * xs
    x3 = x2 * xs
    x4 = x3 * xs
    
    # Trigonometric basis
    sin_x = np.sin(xs)
    cos_x = np.cos(xs)
    sin_2x = np.sin(2.0 * xs)
    
    # Exponential basis (clamped for numerical stability)
    exp_x = np.exp(np.clip(xs, -10, 10))
    exp_neg_x = np.exp(-np.clip(xs, -10, 10))
    
    # Hyperbolic basis
    tanh_x = np.tanh(xs)
    
    # Combine all basis functions
    sinh_term = 0.5 * (exp_x - exp_neg_x)
    
    result = (
        c0 * np.ones_like(x) +
        c1 * xs +
        c2 * x2 +
        c3 * x3 +
        c4 * x4 +
        c5 * sin_x +
        c6 * cos_x +
        c7 * sin_2x +
        c8 * tanh_x +
        c9 * sinh_term
    )
    
    return result
# EVOLVE-BLOCK-END
