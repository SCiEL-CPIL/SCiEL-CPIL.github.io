import numpy as np
import time

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator acceleration model using a pure monomial basis.
    
    This design avoids the linear dependence issues of mixing Chebyshev and monomial
    polynomials, which caused ill-conditioning in previous attempts.
    
    Basis terms (10 parameters):
    0: Constant (bias/driving)
    1: X (linear restoring)
    2: V (linear damping)
    3: X^2 (quadratic restoring, asymmetry)
    4: V^2 (quadratic damping, rectification)
    5: X^3 (cubic restoring, Duffing)
    6: V^3 (cubic damping)
    7: X*V (linear coupling)
    8: X^2*V (Van der Pol damping)
    9: X^3*V (higher-order damping)
    
    Inputs are normalized to approximately [-1, 1] based on training ranges
    to improve numerical conditioning for the BFGS optimizer.
    """
    # Normalize inputs to approximately [-1, 1]
    # Training ranges: x in [-0.5728, 0.5275], v in [-0.1754, 0.2821]
    x_center = -0.02265
    x_scale = 0.55015
    v_center = 0.05335
    v_scale = 0.22875
    
    # Compute normalized variables
    X = (x - x_center) / x_scale
    V = (v - v_center) / v_scale
    
    # Precompute powers
    X2 = X * X
    X3 = X2 * X
    V2 = V * V
    V3 = V2 * V
    
    # Basis terms
    t0 = 1.0
    t1 = X
    t2 = V
    t3 = X2
    t4 = V2
    t5 = X3
    t6 = V3
    t7 = X * V
    t8 = X2 * V
    t9 = X3 * V
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Combine terms
    return (p0 * t0 + p1 * t1 + p2 * t2 + p3 * t3 + p4 * t4 +
            p5 * t5 + p6 * t6 + p7 * t7 + p8 * t8 + p9 * t9)
