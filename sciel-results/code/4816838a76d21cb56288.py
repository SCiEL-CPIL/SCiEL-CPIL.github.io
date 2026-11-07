"""Symbolic regression with 3 inputs - Non-linear parametric model."""
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximate the hidden function using a non-linear parametric model.
    
    Structure:
    h = tanh(a*x + b*y + c*z + d)
    pred = e*h + f*x + g*y + h_*z + i*(x*y + y*z + z*x) + j
    
    This allows for non-linear saturation and interaction effects that
    linear bases (polynomial or trigonometric) cannot easily capture
    without many more parameters.
    """
    p = params.flatten()
    
    if p.size < 10:
        p = np.pad(p, (0, 10 - p.size), mode='constant')
    else:
        p = p[:10]
        
    a, b, c, d, e, f, g, h, i, j = p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9]
    
    # Non-linear hidden unit
    hidden_input = a * x + b * y + c * z + d
    # Clip to avoid overflow in tanh if parameters get large during optimization
    hidden_input = np.clip(hidden_input, -500, 500)
    h_val = np.tanh(hidden_input)
    
    # Interaction term
    interaction = x * y + y * z + z * x
    
    # Combine
    pred = e * h_val + f * x + g * y + h * z + i * interaction + j
    
    return pred
