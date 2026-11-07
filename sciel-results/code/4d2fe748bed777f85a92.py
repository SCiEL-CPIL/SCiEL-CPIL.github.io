"""Rational function with quadratic numerator and quadratic denominator for symbolic regression."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Deadline guard
    start_time = time.time()
    deadline = 145.0  # 5s margin from 150s limit

    # Ensure x and y are 1D arrays for consistent shape handling
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    # Rational function:
    # Numerator: p0 + p1*x + p2*y + p3*x^2 + p4*y^2 + p5*x*y
    # Denominator: 1 + p6*x + p7*y + p8*x^2 + p9*y^2
    #
    # This form can capture:
    # - Polynomial behaviors (when denominator ~ 1)
    # - Saturation/asymptotic behaviors
    # - Pole-like structures
    # - Non-separable smooth functions

    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Compute numerator (quadratic polynomial)
    numerator = (p0 + p1 * x + p2 * y + 
                 p3 * x**2 + p4 * y**2 + p5 * x * y)

    # Compute denominator (1 + quadratic polynomial)
    # The constant term is fixed at 1 to ensure the function is well-defined
    # when all other params are zero, and to avoid a trivial zero denominator
    denominator = (1.0 + p6 * x + p7 * y + 
                   p8 * x**2 + p9 * y**2)

    # Avoid division by zero or near-zero values
    # Use a small epsilon to prevent numerical instability
    eps = 1e-8
    sign_denom = np.sign(denominator)
    abs_denom = np.abs(denominator)
    # Where |denom| < eps, replace with eps * sign (or eps if sign is 0)
    safe_denom = np.where(abs_denom < eps, 
                          eps * np.where(sign_denom == 0, 1.0, sign_denom), 
                          denominator)

    result = numerator / safe_denom

    # Check for non-finite values
    if not np.all(np.isfinite(result)):
        # Return a safe constant if numerical issues arise
        return np.zeros_like(x)

    return result
# EVOLVE-BLOCK-END
