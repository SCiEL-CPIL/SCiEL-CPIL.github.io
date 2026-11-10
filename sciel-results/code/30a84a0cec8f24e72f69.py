"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a product of two low-degree polynomials:
    P_5(x) * Q_4(x), where the 10 parameters define the coefficients.
    
    This creates a non-convex but structured landscape that BFGS can navigate
    to fit functions with sharp local features or high-frequency oscillations
    that linear combinations of fixed smooth basis functions cannot capture.
    
    The product form allows for more complex interactions between polynomial
    terms, enabling richer function approximation while maintaining a manageable
    parameter count.
    """
    x = np.asarray(x, dtype=np.float64)
    p = params
    
    # Extract parameters for P_5(x) and Q_4(x)
    # P_5(x) = p[0] + p[1]*x + p[2]*x^2 + p[3]*x^3 + p[4]*x^4 + p[5]*x^5
    # Q_4(x) = p[6] + p[7]*x + p[8]*x^2 + p[9]*x^3 + 0*x^4 (only 4 params available)
    # We'll use Q_4(x) = p[6] + p[7]*x + p[8]*x^2 + p[9]*x^3
    
    # Compute powers of x
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    x5 = x4 * x
    
    # Compute P_5(x)
    P = p[0] + p[1] * x + p[2] * x2 + p[3] * x3 + p[4] * x4 + p[5] * x5
    
    # Compute Q_4(x) - using 4 parameters for degrees 0-3
    Q = p[6] + p[7] * x + p[8] * x2 + p[9] * x3
    
    # Product
    result = P * Q
    
    # Ensure finite output
    result = np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return result
# EVOLVE-BLOCK-END
