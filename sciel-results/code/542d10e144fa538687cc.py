"""Symbolic regression with 3 inputs using a rich polynomial basis."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression using a complete polynomial basis up to degree 2
    in 3 variables, plus some interaction terms.
    
    The basis includes:
    - Constant: 1
    - Linear: x, y, z
    - Quadratic: x^2, y^2, z^2
    - Interactions: xy, xz, yz
    
    That's 1 + 3 + 3 + 3 = 10 parameters, matching the required length.
    
    This is a richer model than the seed (which only had linear terms + constant),
    allowing it to capture quadratic behavior and interactions.
    """
    p = params
    
    # Ensure finite values
    if not np.all(np.isfinite(p)):
        return np.ones_like(x)
    
    # Build the polynomial basis
    term0 = 1.0                    # constant
    term1 = x                      # x
    term2 = y                      # y
    term3 = z                      # z
    term4 = x * x                  # x^2
    term5 = y * y                  # y^2
    term6 = z * z                  # z^2
    term7 = x * y                  # xy
    term8 = x * z                  # xz
    term9 = y * z                  # yz
    
    result = (p[0] * term0 +
              p[1] * term1 +
              p[2] * term2 +
              p[3] * term3 +
              p[4] * term4 +
              p[5] * term5 +
              p[6] * term6 +
              p[7] * term7 +
              p[8] * term8 +
              p[9] * term9)
    
    return result
# EVOLVE-BLOCK-END
