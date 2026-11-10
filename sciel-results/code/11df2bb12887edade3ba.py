"""Rewritten program for symbolic regression with 1 input."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a product of two low-degree polynomials.
    
    The model is:
    y(x) = P(x) * Q(x)
    where P(x) = p[0] + p[1]*x + p[2]*x^2 + p[3]*x^3 + p[4]*x^4  (degree 4)
         Q(x) = p[5] + p[6]*x + p[7]*x^2 + p[8]*x^3 + p[9]*x^4  (degree 4)
    
    This creates a non-convex but structured landscape that can capture
    sharp local features and high-frequency oscillations that linear
    combinations of fixed smooth basis functions cannot easily represent.
    The product structure allows the model to represent functions with
    varying local behavior through the interaction of the two polynomials.
    """
    x = np.asarray(x, dtype=np.float64)
    p = params
    
    # Compute powers of x
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    
    # Polynomial P(x) with coefficients p[0:5]
    P = p[0] + p[1] * x + p[2] * x2 + p[3] * x3 + p[4] * x4
    
    # Polynomial Q(x) with coefficients p[5:10]
    Q = p[5] + p[6] * x + p[7] * x2 + p[8] * x3 + p[9] * x4
    
    # Product of the two polynomials
    result = P * Q
    
    return result

# EVOLVE-BLOCK-END
