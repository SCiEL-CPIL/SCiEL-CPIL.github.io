"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator acceleration using a polynomial basis in normalized state space.
    
    Parameters:
    - t: time in [30, 50]
    - x: position in [-0.2056, 0.1749]
    - v: velocity in [-0.3391, 0.3624]
    - params: 10 fitted constants
    
    The model uses:
    p0 * 1 + p1 * t_norm + p2 * t_norm^2 + p3 * x + p4 * x^2 + p5 * x^3
    + p6 * v + p7 * v^2 + p8 * v^3 + p9 * x*v
    where t_norm = (t - 40) / 10 to scale t to O(1).
    """
    # Scale t to O(1) magnitude for better BFGS convergence
    t_norm = (t - 40.0) / 10.0
    
    # Compute basis functions
    b0 = np.ones_like(t_norm)
    b1 = t_norm
    b2 = t_norm ** 2
    b3 = x
    b4 = x ** 2
    b5 = x ** 3
    b6 = v
    b7 = v ** 2
    b8 = v ** 3
    b9 = x * v
    
    # Linear combination with fitted parameters
    return (params[0] * b0 +
            params[1] * b1 +
            params[2] * b2 +
            params[3] * b3 +
            params[4] * b4 +
            params[5] * b5 +
            params[6] * b6 +
            params[7] * b7 +
            params[8] * b8 +
            params[9] * b9)
# EVOLVE-BLOCK-END
