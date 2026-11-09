import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator with state-dependent coefficient parameterization.
    Groups nonlinear terms as multiplicative corrections to linear terms
    to improve BFGS conditioning from all-ones initialization.
    
    Form: a*x*(1 + b*x^2) + c*v*(1 + d*v^2) + e*x*v*(1 + f*x + g*v) + h*x^2*v + i*v^3 + j
    """
    p = params
    a, b, c, d, e, f, g, h, i, j = p
    
    x2 = x * x
    v2 = v * v
    xv = x * v
    
    # Linear + cubic correction in x (Duffing-like)
    term_x = a * x * (1.0 + b * x2)
    # Linear + cubic correction in v (nonlinear drag)
    term_v = c * v * (1.0 + d * v2)
    # Cross term with linear corrections
    term_xv = e * xv * (1.0 + f * x + g * v)
    # Additional cubic cross terms
    term_x2v = h * x2 * v
    term_v3 = i * v2 * v
    # Constant
    term_const = j
    
    return term_x + term_v + term_xv + term_x2v + term_v3 + term_const
