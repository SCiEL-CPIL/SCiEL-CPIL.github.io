"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Linear-in-parameters polynomial basis model (GAM-style).
    Uses a fixed set of basis functions for each input and their interactions.
    The 10 parameters are the coefficients for these basis functions.
    
    Basis functions:
    - b: [1, b]
    - s: [1, s, s^2]
    - temp: [1, temp, temp^2]
    - pH: [1, pH, pH^2]
    
    We select 10 key terms from the full tensor product to keep it manageable
    while capturing main effects and some interactions.
    
    Terms:
    1. 1 (bias)
    2. b
    3. s
    4. s^2
    5. temp
    6. temp^2
    7. pH
    8. pH^2
    9. s * temp
    10. s * pH
    """
    # Normalize inputs to improve numerical conditioning
    b_norm = (b - 2.5) / 2.0
    s_norm = (s - 20.0) / 20.0
    temp_norm = (temp - 20.0) / 10.0
    pH_norm = (pH - 7.0) / 5.0

    # Compute basis functions
    one = np.ones_like(b_norm)
    b1 = b_norm
    s1 = s_norm
    s2 = s_norm ** 2
    t1 = temp_norm
    t2 = temp_norm ** 2
    p1 = pH_norm
    p2 = pH_norm ** 2
    s_t = s_norm * temp_norm
    s_p = s_norm * pH_norm

    # Extract parameters
    c0 = params[0]
    c1 = params[1]
    c2 = params[2]
    c3 = params[3]
    c4 = params[4]
    c5 = params[5]
    c6 = params[6]
    c7 = params[7]
    c8 = params[8]
    c9 = params[9]

    # Linear combination
    mu = (c0 * one + 
          c1 * b1 + 
          c2 * s1 + 
          c3 * s2 + 
          c4 * t1 + 
          c5 * t2 + 
          c6 * p1 + 
          c7 * p2 + 
          c8 * s_t + 
          c9 * s_p)

    return mu
# EVOLVE-BLOCK-END
