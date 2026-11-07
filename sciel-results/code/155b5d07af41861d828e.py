"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stable Rational Function Approximation:
    y = (p0 + p1*x + p2*x^2 + p3*x^3 + p4*x^4 + p5*x^5) / (1 + q1^2*x^2 + q2^2*x^4 + q3^2*x^6 + q4^2*x^8)
    
    The denominator is strictly positive (>= 1), ensuring numerical stability and smooth gradients for BFGS.
    This form captures even-symmetric decay/bump behaviors well, while the odd terms in the numerator handle asymmetry.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Extract parameters
    p0, p1, p2, p3, p4, p5 = params[0], params[1], params[2], params[3], params[4], params[5]
    q1, q2, q3, q4 = params[6], params[7], params[8], params[9]
    
    # Precompute powers
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    x5 = x4 * x
    x6 = x3 * x3
    x8 = x4 * x4
    
    # Numerator: Degree 5 polynomial
    numerator = p0 + p1 * x + p2 * x2 + p3 * x3 + p4 * x4 + p5 * x5
    
    # Denominator: 1 + sum of squared terms to ensure positivity and smoothness
    # Using even powers in denominator helps with stability and symmetry
    denominator = 1.0 + (q1 * x)**2 + (q2 * x2)**2 + (q3 * x3)**2 + (q4 * x4)**2
    
    # The denominator is always >= 1, so no division by zero risk
    result = numerator / denominator
    
    # Ensure result is finite (should be, but just in case)
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
# EVOLVE-BLOCK-END
