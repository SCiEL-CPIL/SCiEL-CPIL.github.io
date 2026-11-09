"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a rational function with a bounded denominator.
    This structure is $y = \frac{P(x)}{Q(x)}$ where $P$ is a polynomial and $Q$ is a
    smooth, bounded function derived from a polynomial via tanh. This avoids division
    by zero and allows for flexible non-linear scaling.
    """
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Unpack parameters:
    # a0..a4: coefficients for numerator polynomial (degree 4)
    # b1..b5: coefficients for denominator shaping polynomial (degree 5)
    a0, a1, a2, a3, a4 = params[0], params[1], params[2], params[3], params[4]
    b1, b2, b3, b4, b5 = params[5], params[6], params[7], params[8], params[9]
    
    # Numerator: a0 + a1*x + a2*x^2 + a3*x^3 + a4*x^4
    num = a0 + a1 * x + a2 * x**2 + a3 * x**3 + a4 * x**4
    
    # Denominator shaping polynomial: b1*x + b2*x^2 + b3*x^3 + b4*x^4 + b5*x^5
    # We use tanh to bound this term, ensuring the denominator stays away from zero.
    den_poly = b1 * x + b2 * x**2 + b3 * x**3 + b4 * x**4 + b5 * x**5
    
    # Bounded denominator: 1.5 + 0.5 * tanh(den_poly)
    # Range of tanh is (-1, 1), so den is in (1.0, 2.0).
    # This guarantees numerical stability.
    den = 1.5 + 0.5 * np.tanh(den_poly)
    
    return num / den
# EVOLVE-BLOCK-END
