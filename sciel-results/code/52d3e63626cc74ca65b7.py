"""Symbolic regression: stress-strain of an aluminium rod."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress-strain model for an aluminium rod using a linear-in-parameters basis.
    
    The model uses 10 basis functions that capture:
    - Elastic and plastic strain behavior (polynomial terms)
    - Temperature effects (linear and interaction terms)
    - Strain-temperature coupling
    
    Since the model is linear in params, BFGS converges in 1-2 iterations.
    """
    s = strain
    t = temp
    p = params
    
    # Basis functions:
    # b0: constant
    # b1: strain (elastic)
    # b2: strain^2 (quadratic hardening)
    # b3: strain^3 (cubic hardening)
    # b4: temperature (thermal effect)
    # b5: strain * temp (interaction)
    # b6: strain^2 * temp (interaction)
    # b7: temp^2 (thermal nonlinearity)
    # b8: strain * temp^2 (higher interaction)
    # b9: strain^3 * temp (high-order interaction)
    
    b0 = np.ones_like(s)
    b1 = s
    b2 = s * s
    b3 = s * s * s
    b4 = t
    b5 = s * t
    b6 = s * s * t
    b7 = t * t
    b8 = s * t * t
    b9 = s * s * s * t
    
    return (p[0] * b0 + p[1] * b1 + p[2] * b2 + p[3] * b3 +
            p[4] * b4 + p[5] * b5 + p[6] * b6 + p[7] * b7 +
            p[8] * b8 + p[9] * b9)
# EVOLVE-BLOCK-END
