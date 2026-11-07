"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Linear combination of a diverse set of basis functions.
    The evaluator fits the 10 params via BFGS starting from ones.
    We choose a basis that spans polynomials, trigonometric, sigmoid,
    and localized bump functions to approximate a wide class of targets.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Basis functions (10 total):
    # 0: constant
    # 1: x
    # 2: x^2
    # 3: x^3
    # 4: x^4
    # 5: sin(x)
    # 6: cos(x)
    # 7: tanh(x)
    # 8: exp(-x^2)  (Gaussian bump centered at 0)
    # 9: 1/(1+exp(-4*x))  (steep logistic)
    
    b0 = np.ones_like(x)
    b1 = x
    b2 = x * x
    b3 = x * x * x
    b4 = b2 * b2
    b5 = np.sin(x)
    b6 = np.cos(x)
    b7 = np.tanh(x)
    b8 = np.exp(-b2)
    # Steep logistic to capture step-like behavior
    b9 = 1.0 / (1.0 + np.exp(-4.0 * x))
    
    # Combine with params
    return (params[0] * b0 +
            params[1] * b1 +
            params[2] * b2 +
            params[3] * b3 +
            params[4] * b4 +
            params[5] * b5 +
            params[6] * b6 +
            params[7] * b7 +
            params[8] * b8 +
            params[9] * b9)
# EVOLVE-BLOCK-END
