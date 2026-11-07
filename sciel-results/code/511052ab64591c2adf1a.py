import numpy as np
import time

# EVOLVE-BLOCK-START

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator model: a = f(x, v) with physics-motivated structure.
    
    a = bias + c_x*x + c_x2*x^2 + c_x3*x^3 + c_x5*x^5
        + c_v*v + c_v2*v^2 + c_v3*v^3
        + c_xv*x*v + c_x2v*x^2*v
    
    Scaled to O(1) for BFGS conditioning.
    """
    # Scale inputs to O(1) magnitude
    X = x / 0.2
    V = v / 0.3
    
    # Precompute powers
    X2 = X * X
    X3 = X2 * X
    X5 = X3 * X2
    V2 = V * V
    V3 = V2 * V
    
    # Basis terms (physics-motivated: restoring + damping + nonlinear coupling)
    term_0 = np.ones_like(t)       # bias
    term_1 = X                     # linear restoring
    term_2 = X2                    # quadratic restoring
    term_3 = X3                    # cubic restoring
    term_4 = X5                    # quintic restoring (stiffening/softening)
    term_5 = V                     # linear damping
    term_6 = V2                    # quadratic damping
    term_7 = V3                    # cubic damping
    term_8 = X * V                 # position-velocity coupling
    term_9 = X2 * V                # nonlinear drag
    
    # Combine with parameters
    pred = (params[0] * term_0 +
            params[1] * term_1 +
            params[2] * term_2 +
            params[3] * term_3 +
            params[4] * term_4 +
            params[5] * term_5 +
            params[6] * term_6 +
            params[7] * term_7 +
            params[8] * term_8 +
            params[9] * term_9)
    
    return pred

# EVOLVE-BLOCK-END
