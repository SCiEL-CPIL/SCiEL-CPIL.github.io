"""Symbolic regression with Rational Function Approximation (Padé-like)."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Rational function approximation using a ratio of two polynomials.
    
    y(x) = P(x) / Q(x)
    P(x) = a0 + a1*x + a2*x^2 + a3*x^3 + a4*x^4
    Q(x) = 1 + b1*x + b2*x^2 + b3*x^3 + b4*x^4
    
    params (length 10):
    params[0:5] are a0..a4
    params[5:9] are b1..b4
    The constant term of Q is fixed to 1 to ensure stability near x=0.
    
    This form can capture non-polynomial behaviors (asymptotes, hyperbolic, etc.) 
    more efficiently than pure polynomials of the same degree.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Extract parameters
    a = params[0:5]
    b = params[5:9]
    
    # Compute powers of x
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    
    # Numerator P(x) = a0 + a1*x + a2*x^2 + a3*x^3 + a4*x^4
    # Using Horner's rule for efficiency and numerical stability
    P = a[4]
    P = P * x + a[3]
    P = P * x + a[2]
    P = P * x + a[1]
    P = P * x + a[0]
    
    # Denominator Q(x) = 1 + b1*x + b2*x^2 + b3*x^3 + b4*x^4
    # Note: We fix the constant term to 1.
    Q = b[3]
    Q = Q * x + b[2]
    Q = Q * x + b[1]
    Q = Q * x + b[0]
    Q = Q * x + 1.0  # Add the fixed constant 1
    
    # Safety check: Prevent division by zero or near-zero values
    # Clip the denominator to avoid singularities
    # We preserve the sign to maintain the function's shape
    eps = 1e-6
    Q_safe = np.where(np.abs(Q) < eps, np.sign(Q) * eps, Q)
    # If Q is exactly 0, sign is 0, so we default to eps
    Q_safe = np.where(Q == 0, eps, Q_safe)
    
    # Calculate result
    result = P / Q_safe
    
    # Ensure finite values (handle any overflow or inf)
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
# EVOLVE-BLOCK-END
