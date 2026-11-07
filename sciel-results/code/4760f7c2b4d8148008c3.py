"""Nonlinear oscillator symbolic regression with physics-informed basis."""
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a damped nonlinear oscillator (Duffing-type) with
    physics-informed basis terms.
    
    Basis: {x, x^3, v, v|v|, x^2*v, x|v|, sin(x), sin(3x), v*sin(x), 1}
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Position terms
    x2 = x * x
    x3 = x2 * x
    
    # Velocity terms
    abs_v = np.abs(v)
    v_abs_v = v * abs_v  # v|v|
    
    # Interaction terms
    x2_v = x2 * v
    x_abs_v = x * abs_v  # x|v|
    
    # Trigonometric terms
    sin_x = np.sin(x)
    sin_3x = np.sin(3.0 * x)
    v_sin_x = v * sin_x
    
    # Combine
    result = (
        p0 * x +
        p1 * x3 +
        p2 * v +
        p3 * v_abs_v +
        p4 * x2_v +
        p5 * x_abs_v +
        p6 * sin_x +
        p7 * sin_3x +
        p8 * v_sin_x +
        p9
    )
    
    return result
