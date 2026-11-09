"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Combined polynomial + sigmoid basis for symbolic regression.
    
    Basis:
    0: constant (1)
    1: x
    2: x^2
    3: x^3
    4: sigmoid(x) = 1/(1+exp(-x))
    5: sigmoid(2x)
    6: sigmoid(x-1)
    7: sigmoid(x+1)
    8: exp(-x^2)  (Gaussian bump)
    9: exp(-(x-1)^2)  (offset Gaussian)
    
    This hybrid captures both global polynomial trends and local
    nonlinear transitions, suitable for a wide range of hidden functions.
    """
    p = params
    x = np.asarray(x, dtype=np.float64)
    
    # Polynomial terms
    x2 = x * x
    x3 = x2 * x
    
    # Sigmoid terms (stable computation)
    def sigmoid(z):
        # Use np.where to avoid overflow in exp
        return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))
    
    sig_x = sigmoid(x)
    sig_2x = sigmoid(2.0 * x)
    sig_xm1 = sigmoid(x - 1.0)
    sig_xp1 = sigmoid(x + 1.0)
    
    # Gaussian bump terms
    gauss0 = np.exp(-x2)
    gauss1 = np.exp(-(x - 1.0) ** 2)
    
    # Linear combination
    result = (p[0] +
              p[1] * x + p[2] * x2 + p[3] * x3 +
              p[4] * sig_x + p[5] * sig_2x +
              p[6] * sig_xm1 + p[7] * sig_xp1 +
              p[8] * gauss0 + p[9] * gauss1)
    
    return result
# EVOLVE-BLOCK-END
