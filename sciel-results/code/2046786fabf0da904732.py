"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression model using a mixed basis of polynomials, exponentials, and logarithms.
    
    The basis is designed to be well-conditioned for BFGS optimization starting from all-ones.
    x is in [0.1, 6]. We use a combination of:
    - Constant term
    - Linear and quadratic polynomial terms
    - Exponential decay terms (exp(-a*x) for various a)
    - Logarithmic growth (log1p(x))
    - Reciprocal term (1/(1+x))
    
    This provides flexibility to capture exponential decay, logarithmic growth,
    and polynomial trends common in hidden target functions.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        x = x.ravel()
    
    p = params
    
    # Compute basis functions
    b0 = 1.0
    b1 = x
    b2 = x ** 2
    b3 = np.exp(-0.5 * x)
    b4 = np.exp(-1.0 * x)
    b5 = np.exp(-2.0 * x)
    b6 = np.log1p(x)
    b7 = 1.0 / (1.0 + x)
    b8 = x ** 3
    b9 = np.exp(-0.25 * x)
    
    # Linear combination
    y = (p[0] * b0 + 
         p[1] * b1 + 
         p[2] * b2 + 
         p[3] * b3 + 
         p[4] * b4 + 
         p[5] * b5 + 
         p[6] * b6 + 
         p[7] * b7 + 
         p[8] * b8 + 
         p[9] * b9)
    
    return y
# EVOLVE-BLOCK-END
