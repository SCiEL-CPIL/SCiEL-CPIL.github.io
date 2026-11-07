"""Symbolic regression with 1 input(s) - Flexible sigmoid units with learnable centers/slopes."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Model: Linear combination of 10 basis functions using flexible sigmoid units
    with learnable centers and slopes, plus a Gaussian bump and polynomial terms.
    
    The 10 params are used as:
      0: weight for constant
      1: weight for x
      2: weight for sigmoid unit 1 (center = 0, slope = 2)
      3: weight for sigmoid unit 2 (center = 1, slope = 3)
      4: weight for sigmoid unit 3 (center = -1, slope = 3)
      5: weight for sigmoid unit 4 (center = 0.5, slope = 5)
      6: weight for sigmoid unit 5 (center = -0.5, slope = 5)
      7: weight for Gaussian bump exp(-x^2)
      8: weight for x^2
      9: weight for x^3
    
    Fixed centers/slopes provide good conditioning while the weights allow
    BFGS to adapt. This design captures localized non-linear features and
    asymptotic behavior better than fixed orthogonal bases.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    p = params
    
    # Clip x to avoid numerical issues
    x_c = np.clip(x, -50, 50)
    
    # Basis functions
    b0 = np.ones_like(x_c)
    b1 = x_c
    
    # Flexible sigmoid units with fixed centers and slopes
    # sigmoid(z) = 1 / (1 + exp(-z))
    def sigmoid(z):
        # Numerically stable sigmoid
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))
    
    # Sigmoid unit 1: center=0, slope=2 -> sigmoid(2*x)
    b2 = sigmoid(2.0 * x_c)
    
    # Sigmoid unit 2: center=1, slope=3 -> sigmoid(3*(x-1))
    b3 = sigmoid(3.0 * (x_c - 1.0))
    
    # Sigmoid unit 3: center=-1, slope=3 -> sigmoid(3*(x+1))
    b4 = sigmoid(3.0 * (x_c + 1.0))
    
    # Sigmoid unit 4: center=0.5, slope=5 -> sigmoid(5*(x-0.5))
    b5 = sigmoid(5.0 * (x_c - 0.5))
    
    # Sigmoid unit 5: center=-0.5, slope=5 -> sigmoid(5*(x+0.5))
    b6 = sigmoid(5.0 * (x_c + 0.5))
    
    # Gaussian bump
    exp_arg = -x_c ** 2
    exp_arg = np.maximum(exp_arg, -100)
    b7 = np.exp(exp_arg)
    
    # Polynomial terms
    b8 = x_c ** 2
    b9 = x_c ** 3
    
    # Linear combination
    y = (p[0] * b0 + p[1] * b1 + p[2] * b2 + p[3] * b3 + p[4] * b4 +
         p[5] * b5 + p[6] * b6 + p[7] * b7 +
         p[8] * b8 + p[9] * b9)
    
    return y
# EVOLVE-BLOCK-END
