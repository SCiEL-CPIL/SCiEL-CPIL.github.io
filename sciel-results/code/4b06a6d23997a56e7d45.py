"""Symbolic regression with 1 input - Linear basis: polynomial + trig + localized."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Linear-in-parameters model with 10 fixed basis functions.
    
    Basis:
      0: 1              (constant)
      1: x              (linear)
      2: x^2            (quadratic)
      3: x^3            (cubic)
      4: x^4            (quartic)
      5: sin(x)         (periodic)
      6: cos(x)         (periodic)
      7: exp(-x^2)      (Gaussian bump)
      8: 1/(1+x^2)      (Lorentzian)
      9: x * exp(-x^2)  (odd localized)
    
    y = sum_{i=0}^{9} params[i] * phi_i(x)
    
    Since the model is linear in params, BFGS converges to the global
    least-squares optimum in one or two steps from any initialization,
    avoiding the local-minima traps of non-linear architectures.
    """
    p = params
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    exp_neg_x2 = np.exp(-x2)
    
    phi0 = np.ones_like(x)
    phi1 = x
    phi2 = x2
    phi3 = x3
    phi4 = x4
    phi5 = np.sin(x)
    phi6 = np.cos(x)
    phi7 = exp_neg_x2
    phi8 = 1.0 / (1.0 + x2)
    phi9 = x * exp_neg_x2
    
    y = (p[0] * phi0 + p[1] * phi1 + p[2] * phi2 + p[3] * phi3 +
         p[4] * phi4 + p[5] * phi5 + p[6] * phi6 + p[7] * phi7 +
         p[8] * phi8 + p[9] * phi9)
    
    return y
# EVOLVE-BLOCK-END
