"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
_TIME_LIMIT = 150.0
_TIME_MARGIN = 5.0  # safety margin in seconds


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Complete quadratic basis on normalized inputs.
    
    Structural rationale:
    - The full quadratic polynomial in 3 variables has exactly 10 terms:
      1, x, y, z, x^2, y^2, z^2, xy, xz, yz
    - This provides the richest smooth approximation possible with 10 parameters
      if the target is a low-degree polynomial or can be well-approximated by one.
    - Normalization to [-1, 1] improves conditioning for BFGS starting from all ones,
      ensuring that the quadratic terms (which would have larger magnitude in [0.5, 2])
      are on a similar scale to linear terms.
    
    Basis allocation (10 params):
    p0: bias
    p1: xn (normalized x)
    p2: yn (normalized y)
    p3: zn (normalized z)
    p4: xn^2
    p5: yn^2
    p6: zn^2
    p7: xn*yn
    p8: xn*zn
    p9: yn*zn
    """
    # Check time budget
    if time.time() - _START_TIME > _TIME_LIMIT - _TIME_MARGIN:
        # Fallback: return a simple prediction using only bias
        return np.full_like(np.asarray(x, dtype=float), params[0], dtype=float)
    
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    
    p = np.asarray(params, dtype=float)
    
    # Normalize inputs to [-1, 1] range
    # Original range is [0.5, 2], center is 1.25, half-range is 0.75
    xn = (x - 1.25) / 0.75
    yn = (y - 1.25) / 0.75
    zn = (z - 1.25) / 0.75
    
    # Build the complete quadratic basis
    b0 = np.ones_like(xn)
    b1 = xn
    b2 = yn
    b3 = zn
    b4 = xn * xn
    b5 = yn * yn
    b6 = zn * zn
    b7 = xn * yn
    b8 = xn * zn
    b9 = yn * zn
    
    # Prediction as linear combination of basis
    pred = (p[0] * b0 +
            p[1] * b1 +
            p[2] * b2 +
            p[3] * b3 +
            p[4] * b4 +
            p[5] * b5 +
            p[6] * b6 +
            p[7] * b7 +
            p[8] * b8 +
            p[9] * b9)
    
    return pred
# EVOLVE-BLOCK-END
