"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator model with polynomial terms in x, v and periodic terms in t.
    Basis functions:
    0: 1
    1: x
    2: v
    3: x^2
    4: x^3
    5: cos(t)
    6: sin(t)
    7: cos(2t)
    8: sin(2t)
    9: x*v
    """
    p = params
    term0 = p[0]
    term1 = p[1] * x
    term2 = p[2] * v
    term3 = p[3] * x**2
    term4 = p[4] * x**3
    term5 = p[5] * np.cos(t)
    term6 = p[6] * np.sin(t)
    term7 = p[7] * np.cos(2 * t)
    term8 = p[8] * np.sin(2 * t)
    term9 = p[9] * x * v
    
    return term0 + term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9
# EVOLVE-BLOCK-END
