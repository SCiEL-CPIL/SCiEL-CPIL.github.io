"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a product of two low-degree polynomials.
    
    The model is P(x) * Q(x) where P is degree 3 (4 params) and Q is degree 2 (3 params),
    plus an additional linear term with 3 params to capture additive components.
    This creates a non-convex but structured landscape that can fit functions with
    sharper features than a pure linear basis.
    """
    x = np.asarray(x, dtype=np.float64)
    p = params
    
    # Extract parameters
    # p[0:4] -> P coefficients (degree 0..3)
    # p[4:7] -> Q coefficients (degree 0..2)
    # p[7:10] -> additive linear term coefficients (degree 0..2)
    
    a0, a1, a2, a3 = p[0], p[1], p[2], p[3]
    b0, b1, b2 = p[4], p[5], p[6]
    c0, c1, c2 = p[7], p[8], p[9]
    
    # Compute P(x) = a0 + a1*x + a2*x^2 + a3*x^3
    x2 = x * x
    x3 = x2 * x
    P = a0 + a1 * x + a2 * x2 + a3 * x3
    
    # Compute Q(x) = b0 + b1*x + b2*x^2
    Q = b0 + b1 * x + b2 * x2
    
    # Additive term: c0 + c1*x + c2*x^2
    add = c0 + c1 * x + c2 * x2
    
    return P * Q + add

# EVOLVE-BLOCK-END
