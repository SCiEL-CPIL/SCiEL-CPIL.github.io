"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Product-of-Sums decomposition for symbolic regression.
    
    Model: f(x, y) = P(x) * Q(y)
    where P(x) = p0 + p1*x + p2*x^2 + p3*x^3 + p4*x^4
          Q(y) = p5 + p6*y + p7*y^2 + p8*y^3 + p9*y^4
    
    This structure captures multiplicative interactions between x and y
    efficiently, which linear additive models often fail to do with limited parameters.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    
    # Unpack parameters
    p0, p1, p2, p3, p4 = params[0], params[1], params[2], params[3], params[4]
    p5, p6, p7, p8, p9 = params[5], params[6], params[7], params[8], params[9]
    
    # Compute polynomial components for x
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    px = p0 + p1 * x + p2 * x2 + p3 * x3 + p4 * x4
    
    # Compute polynomial components for y
    y2 = y * y
    y3 = y2 * y
    y4 = y2 * y2
    py = p5 + p6 * y + p7 * y2 + p8 * y3 + p9 * y4
    
    # Product of sums
    return px * py
# EVOLVE-BLOCK-END
