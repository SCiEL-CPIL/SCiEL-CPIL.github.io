# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Acceleration of a damped nonlinear oscillator with driving force, from position and velocity."""
    # Ensure inputs are 1D arrays for consistent output shape
    x = np.asarray(x, dtype=np.float64).ravel()
    v = np.asarray(v, dtype=np.float64).ravel()
    p = np.asarray(params, dtype=np.float64)
    
    # Compute basis terms
    x2 = x * x
    x3 = x2 * x
    v2 = v * v
    v3 = v2 * v
    x2v = x2 * v
    xv2 = x * v2
    x4 = x2 * x2
    v4 = v2 * v2
    x3v = x3 * v
    xv3 = x * v3
    
    # 10-parameter sparse basis for nonlinear oscillator
    # Dominant physics terms:
    # p[0]*x + p[1]*x^3 (nonlinear spring)
    # p[2]*v + p[3]*v^3 (nonlinear damping)
    # p[4]*x^2*v + p[5]*x*v^2 (cross terms)
    # p[6]*x^4 + p[7]*v^4 (higher order)
    # p[8]*x^3*v + p[9]*x*v^3 (higher cross)
    
    result = (p[0] * x +
              p[1] * x3 +
              p[2] * v +
              p[3] * v3 +
              p[4] * x2v +
              p[5] * xv2 +
              p[6] * x4 +
              p[7] * v4 +
              p[8] * x3v +
              p[9] * xv3)
    
    return result
# EVOLVE-BLOCK-END
