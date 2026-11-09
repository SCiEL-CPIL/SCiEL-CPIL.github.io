"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression using a stable basis set.
    Basis: 1, x, x^2, x^3, sin(x), cos(x), tanh(x), exp(-x^2), sin(2x), cos(2x).
    This avoids high-degree polynomial instability while covering common functional forms.
    """
    x = np.asarray(x, dtype=np.float64)
    
    p0 = params[0]
    p1 = params[1]
    p2 = params[2]
    p3 = params[3]
    p4 = params[4]
    p5 = params[5]
    p6 = params[6]
    p7 = params[7]
    p8 = params[8]
    p9 = params[9]
    
    # Polynomial terms
    x2 = x * x
    x3 = x2 * x
    
    # Trigonometric terms
    sin_x = np.sin(x)
    cos_x = np.cos(x)
    sin_2x = np.sin(2 * x)
    cos_2x = np.cos(2 * x)
    
    # Hyperbolic and Exponential terms
    tanh_x = np.tanh(x)
    exp_neg_x2 = np.exp(-x2)
    
    y = (p0 * np.ones_like(x) +
         p1 * x +
         p2 * x2 +
         p3 * x3 +
         p4 * sin_x +
         p5 * cos_x +
         p6 * tanh_x +
         p7 * exp_neg_x2 +
         p8 * sin_2x +
         p9 * cos_2x)
    
    return y
# EVOLVE-BLOCK-END
