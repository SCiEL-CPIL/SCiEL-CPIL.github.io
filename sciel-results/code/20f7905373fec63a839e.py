import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Rational Chebyshev model: f(x) = N(x) / D(x)
    where N and D are degree-4 Chebyshev polynomials.
    
    This rational form can capture singularities, asymptotic behaviors,
    and complex non-polynomial curvatures that pure polynomials or
    Gaussian+poly hybrids cannot, using all 10 parameters productively.
    
    Params (10 total):
    0-4: a0..a4 - Chebyshev coefficients for numerator
    5-9: b0..b4 - Chebyshev coefficients for denominator
    
    To ensure stability, the denominator is constrained to be positive
    near the origin by adding a large constant floor.
    """
    a0, a1, a2, a3, a4 = params[0], params[1], params[2], params[3], params[4]
    b0, b1, b2, b3, b4 = params[5], params[6], params[7], params[8], params[9]
    
    # Compute Chebyshev polynomials T0..T4 for input x
    # T0 = 1
    # T1 = x
    # T2 = 2x^2 - 1
    # T3 = 4x^3 - 3x
    # T4 = 8x^4 - 8x^2 + 1
    T0 = np.ones_like(x)
    T1 = x
    T2 = 2.0 * x * x - 1.0
    T3 = 4.0 * x * x * x - 3.0 * x
    T4 = 8.0 * x * x * x * x - 8.0 * x * x + 1.0
    
    # Numerator
    N = a0 * T0 + a1 * T1 + a2 * T2 + a3 * T3 + a4 * T4
    
    # Denominator with a floor to prevent division by zero and ensure
    # the denominator doesn't cross zero in [-2, 2] for typical parameter values.
    # Starting from all ones, D = 5, which is safe.
    D = b0 * T0 + b1 * T1 + b2 * T2 + b3 * T3 + b4 * T4
    
    # Add a large constant to prevent the denominator from approaching zero,
    # which would cause numerical instability and non-finite values.
    # The constant 10.0 is large enough to keep D well away from 0 for
    # reasonable parameter magnitudes while still allowing the rational
    # structure to shape the function.
    D = D + 10.0
    
    # Safety: clip D to avoid division by very small numbers
    D_safe = np.where(np.abs(D) < 0.1, np.sign(D) * 0.1 + (D == 0) * 0.1, D)
    
    y = N / D_safe
    
    # Ensure finiteness
    y = np.where(np.isfinite(y), y, 0.0)
    
    return y
