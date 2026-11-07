"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Bivariate polynomial expansion for stress-strain of an aluminium rod.
    
    Uses a full polynomial basis up to degree 3 in strain and degree 2 in temp,
    truncated to use exactly 10 parameters.
    
    Basis functions:
    1. 1
    2. strain
    3. temp
    4. strain^2
    5. strain*temp
    6. temp^2
    7. strain^3
    8. strain^2*temp
    9. strain*temp^2
    10. temp^3
    
    This captures non-linear interactions and curvature in both variables.
    """
    e = strain
    t = temp
    p = params
    
    # Polynomial terms
    term1 = np.ones_like(e)
    term2 = e
    term3 = t
    term4 = e * e
    term5 = e * t
    term6 = t * t
    term7 = e * e * e
    term8 = e * e * t
    term9 = e * t * t
    term10 = t * t * t
    
    return (p[0] * term1 +
            p[1] * term2 +
            p[2] * term3 +
            p[3] * term4 +
            p[4] * term5 +
            p[5] * term6 +
            p[6] * term7 +
            p[7] * term8 +
            p[8] * term9 +
            p[9] * term10)
# EVOLVE-BLOCK-END
