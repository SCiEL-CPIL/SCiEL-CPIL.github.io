"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_TIMEOUT_MARGIN = 148.0  # seconds, keep a safety margin before 150s kill


def _time_left():
    return _start_time + _TIMEOUT_MARGIN - time.time()


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression with 10 parameters that tries to recover an unknown function.
    Strategy: Bivariate Polynomial Basis of Total Degree 3.
    
    The basis functions are:
    1. 1
    2. x
    3. y
    4. x^2
    5. xy
    6. y^2
    7. x^3
    8. x^2y
    9. xy^2
    10. y^3
    
    This provides a smooth (C-infinity) global approximation surface.
    Unlike piecewise linear or low-degree Chebyshev, this captures
    cubic interactions and trends across the entire [-1, 1]^2 domain
    with exactly 10 parameters, offering a well-conditioned, smooth
    loss landscape for BFGS optimization.
    """
    # Ensure inputs are 1D arrays
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    params = np.asarray(params, dtype=np.float64)
    
    # Check for timeout just in case, though this is a vectorized op
    if _time_left() < 0:
        return np.zeros_like(x)

    # Extract parameters
    # p0: const
    # p1: x
    # p2: y
    # p3: x^2
    # p4: xy
    # p5: y^2
    # p6: x^3
    # p7: x^2 y
    # p8: x y^2
    # p9: y^3
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Compute powers
    x2 = x * x
    x3 = x2 * x
    y2 = y * y
    y3 = y2 * y
    
    # Construct prediction
    # pred = p0 + p1*x + p2*y + p3*x^2 + p4*x*y + p5*y^2 + p6*x^3 + p7*x^2*y + p8*x*y^2 + p9*y^3
    
    pred = p0
    pred += p1 * x
    pred += p2 * y
    pred += p3 * x2
    pred += p4 * x * y
    pred += p5 * y2
    pred += p6 * x3
    pred += p7 * x2 * y
    pred += p8 * x * y2
    pred += p9 * y3
    
    return pred
# EVOLVE-BLOCK-END
