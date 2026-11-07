"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a damped nonlinear oscillator with driving force, from position and velocity.
    
    Uses a Fourier series basis in the bounded phase-space coordinates (x, v) to model
    periodic dynamics. The basis includes fundamental and harmonic terms in x, v, and
    their combinations, allowing efficient capture of nonlinear couplings.
    
    Basis functions (10 terms):
      1. sin(x)
      2. cos(x)
      3. sin(v)
      4. cos(v)
      5. sin(x + v)
      6. cos(x + v)
      7. sin(x - v)
      8. cos(x - v)
      9. sin(2*x)
      10. cos(2*x)
    """
    # Precompute basis functions
    sin_x = np.sin(x)
    cos_x = np.cos(x)
    sin_v = np.sin(v)
    cos_v = np.cos(v)
    
    sum_xv = x + v
    diff_xv = x - v
    sin_sum = np.sin(sum_xv)
    cos_sum = np.cos(sum_xv)
    sin_diff = np.sin(diff_xv)
    cos_diff = np.cos(diff_xv)
    
    sin_2x = np.sin(2.0 * x)
    cos_2x = np.cos(2.0 * x)
    
    # Combine with parameters
    p = params
    result = (
        p[0] * sin_x +
        p[1] * cos_x +
        p[2] * sin_v +
        p[3] * cos_v +
        p[4] * sin_sum +
        p[5] * cos_sum +
        p[6] * sin_diff +
        p[7] * cos_diff +
        p[8] * sin_2x +
        p[9] * cos_2x
    )
    
    return result
# EVOLVE-BLOCK-END
