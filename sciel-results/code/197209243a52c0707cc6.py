# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator using a hybrid basis:
    - Trigonometric terms (sin, cos) for oscillatory behavior
    - Polynomial terms for nonlinear corrections
    - Interaction terms between position and velocity
    
    This structural change moves away from pure polynomial basis to capture
    the inherent periodic nature of oscillator dynamics.
    
    Inputs:
        x: position, range ~[-0.57, 0.53]
        v: velocity, range ~[-0.18, 0.28]
        params: array of 10 parameters
    
    Returns:
        acceleration array, one prediction per row
    """
    p = params
    
    # Scale inputs to reasonable ranges for trig functions
    # x: [-0.5728, 0.5275] -> scale to ~[-pi, pi] for sin/cos
    # v: [-0.1754, 0.2821] -> scale to ~[-pi, pi]
    X = x * 5.0   # brings x to ~[-2.86, 2.64], within [-pi, pi]
    V = v * 10.0  # brings v to ~[-1.75, 2.82], within [-pi, pi]
    
    # Precompute common terms
    X2 = X * X
    V2 = V * V
    X3 = X2 * X
    
    # Basis functions:
    # 1. sin(X) - primary oscillatory term in position
    # 2. cos(X) - constant offset from position
    # 3. sin(V) - oscillatory term in velocity
    # 4. X - linear position correction
    # 5. X^3 - cubic nonlinear position term
    # 6. V - linear velocity term
    # 7. X*V - position-velocity interaction
    # 8. sin(X)*V - modulated velocity by position phase
    # 9. cos(X)*V - modulated velocity by position phase (quadrature)
    # 10. 1 - bias term
    
    b1 = np.sin(X)
    b2 = np.cos(X)
    b3 = np.sin(V)
    b4 = X
    b5 = X3
    b6 = V
    b7 = X * V
    b8 = np.sin(X) * V
    b9 = np.cos(X) * V
    b10 = np.ones_like(X)
    
    # Combine with parameters
    y = (p[0] * b1 +
         p[1] * b2 +
         p[2] * b3 +
         p[3] * b4 +
         p[4] * b5 +
         p[5] * b6 +
         p[6] * b7 +
         p[7] * b8 +
         p[8] * b9 +
         p[9] * b10)
    
    return y
# EVOLVE-BLOCK-END
