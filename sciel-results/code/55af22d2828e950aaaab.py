"""Symbolic regression with rational function approximation."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Rational function approximation:
    y = (p0 + p1*x + p2*x^2 + p3*x^3 + p4*x^4 + p5*x^5 + p6*x^6 + p7*x^7)
        / (1 + a*x + b*x^2)
    
    params layout (length 10):
      params[0..7] = numerator coefficients p0..p7
      params[8]    = denominator linear coeff (bounded via tanh)
      params[9]    = denominator quadratic coeff (bounded via tanh)
    
    Denominator: 1 + 0.2*tanh(params[8])*x + 0.1*tanh(params[9])*x^2
    Worst case over [-2,2]: 1 - 0.2*2 - 0.1*4 = 1 - 0.4 - 0.4 = 0.2 > 0
    This ensures the denominator is always positive, keeping BFGS stable.
    """
    x = np.asarray(x, dtype=float)
    
    # Extract parameters
    p0 = params[0]
    p1 = params[1]
    p2 = params[2]
    p3 = params[3]
    p4 = params[4]
    p5 = params[5]
    p6 = params[6]
    p7 = params[7]
    q0 = params[8]
    q1 = params[9]
    
    # Compute powers of x
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    x5 = x4 * x
    x6 = x3 * x3
    x7 = x6 * x
    
    # Numerator: 7th degree polynomial
    numerator = (p0 + p1 * x + p2 * x2 + p3 * x3 +
                 p4 * x4 + p5 * x5 + p6 * x6 + p7 * x7)
    
    # Denominator with bounded coefficients ensuring positivity over [-2,2]
    # 1 + 0.2*tanh(q0)*x + 0.1*tanh(q1)*x^2
    # Min value: 1 - 0.2*2 - 0.1*4 = 0.2 > 0
    a = 0.2 * np.tanh(q0)
    b = 0.1 * np.tanh(q1)
    
    denominator = 1.0 + a * x + b * x2
    
    # Safety: clip denominator to avoid division by zero (shouldn't trigger)
    denominator = np.where(np.abs(denominator) < 1e-8, 1e-8, denominator)
    
    result = numerator / denominator
    
    # Ensure finite output
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
# EVOLVE-BLOCK-END
