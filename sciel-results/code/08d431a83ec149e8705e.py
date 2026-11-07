"""Symbolic regression: 10-parameter hybrid basis for 2D function recovery."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 140.0  # Safety margin before 150s limit


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    10-parameter hybrid basis:
    - Polynomial terms up to degree 3 (excluding some high-order interactions)
    - Hyperbolic tangent terms for non-polynomial shape capture
    
    Features:
      0: 1
      1: x
      2: y
      3: x^2
      4: y^2
      5: xy
      6: x^3
      7: y^3
      8: tanh(x + y)
      9: tanh(x - y)
    """
    # Deadline guard
    if time.time() > _DEADLINE:
        return np.zeros_like(x, dtype=float)

    p = np.asarray(params, dtype=float)
    
    # Polynomial terms
    T1x = x
    T1y = y
    T2x = x * x
    T2y = y * y
    T3x = x * x * x
    T3y = y * y * y
    Txy = x * y
    
    # Non-polynomial terms
    # tanh is smooth and bounded, good for capturing saturation or sigmoid shapes
    # It is approximated by z - z^3/3 + ... so it overlaps with polynomial space
    # but extends the representable function class.
    T_tanh_sum = np.tanh(x + y)
    T_tanh_diff = np.tanh(x - y)
    
    pred = (
        p[0] +
        p[1] * T1x +
        p[2] * T1y +
        p[3] * T2x +
        p[4] * T2y +
        p[5] * Txy +
        p[6] * T3x +
        p[7] * T3y +
        p[8] * T_tanh_sum +
        p[9] * T_tanh_diff
    )
    
    return pred
# EVOLVE-BLOCK-END
