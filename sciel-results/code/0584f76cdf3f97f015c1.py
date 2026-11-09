# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form acceleration for a damped nonlinear oscillator with driving.
    Uses normalized inputs for better conditioning and a physics-inspired basis.
    
    Inputs:
        t: time (range [30, 50])
        x: position (range [-0.2056, 0.1749])
        v: velocity (range [-0.3391, 0.3624])
        params: array of 10 fitted constants
    
    Returns:
        acceleration (one per row)
    """
    # Normalize inputs to improve conditioning
    t_norm = (t - 40.0) / 10.0       # centered around midpoint, scaled to ~[-1, 1]
    x_norm = x / 0.2                  # scale to ~[-1, 1]
    v_norm = v / 0.35                 # scale to ~[-1, 1]
    
    p = params
    
    # Basis functions:
    # Linear terms
    b0 = np.ones_like(x_norm)
    b1 = x_norm
    b2 = v_norm
    b3 = t_norm
    
    # Nonlinear spring/damping terms
    b4 = x_norm ** 3          # cubic spring
    b5 = v_norm * np.abs(v_norm)  # cubic damping (odd)
    b6 = x_norm * v_norm       # cross term
    b7 = x_norm ** 2           # quadratic spring
    b8 = v_norm ** 2           # quadratic damping
    b9 = np.sin(2.0 * np.pi * t_norm)  # periodic drive component
    
    # Linear combination
    result = (p[0] * b0 + p[1] * b1 + p[2] * b2 + p[3] * b3 +
              p[4] * b4 + p[5] * b5 + p[6] * b6 + p[7] * b7 +
              p[8] * b8 + p[9] * b9)
    
    return result
# EVOLVE-BLOCK-END
