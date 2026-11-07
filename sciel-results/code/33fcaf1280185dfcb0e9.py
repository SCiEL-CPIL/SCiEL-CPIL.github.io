"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 140.0

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator.
    Uses a combination of polynomial and nonlinear (sin, tanh) terms to capture
    both weakly and strongly nonlinear behavior.
    
    Parameters (10 total):
    p[0]: constant offset
    p[1]: linear x coefficient
    p[2]: linear v coefficient
    p[3]: x^2 coefficient
    p[4]: x*v coefficient
    p[5]: v^2 coefficient
    p[6]: sin(x) coefficient
    p[7]: tanh(x) coefficient
    p[8]: x^3 coefficient
    p[9]: v^3 coefficient
    """
    p = params
    
    # Precompute powers
    x2 = x * x
    x3 = x2 * x
    v2 = v * v
    v3 = v2 * v
    
    # Nonlinear terms
    sin_x = np.sin(x)
    tanh_x = np.tanh(x)
    
    # Combine all terms
    return (
        p[0] +
        p[1] * x +
        p[2] * v +
        p[3] * x2 +
        p[4] * x * v +
        p[5] * v2 +
        p[6] * sin_x +
        p[7] * tanh_x +
        p[8] * x3 +
        p[9] * v3
    )
# EVOLVE-BLOCK-END
