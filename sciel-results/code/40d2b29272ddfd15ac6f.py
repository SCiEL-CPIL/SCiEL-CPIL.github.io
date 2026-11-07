"""Symbolic regression: nonlinear oscillator (2) - Physics-Informed Basis."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator.
    
    Model:
    y = sum_{i=0}^{9} p_i * b_i(t, x, v)
    
    Basis functions are chosen to match common nonlinear oscillator dynamics
    (Duffing/Van der Pol type) with periodic driving.
    Inputs are normalized to improve numerical conditioning for BFGS.
    
    Basis:
    0: 1 (Bias)
    1: X (Linear restoring)
    2: V (Linear damping)
    3: X^3 (Cubic nonlinearity)
    4: X^2 * V (Nonlinear damping / parametric)
    5: sin(t) (Driving in phase)
    6: cos(t) (Driving quadrature)
    7: X * sin(t) (State-dependent forcing)
    8: X * cos(t) (State-dependent forcing)
    9: V * cos(t) (Velocity-dependent forcing)
    """
    # Normalize x and v to O(1) range to help optimizer
    # x range approx [-0.2, 0.2] -> scale by 0.2
    # v range approx [-0.35, 0.35] -> scale by 0.35
    X = x / 0.2
    V = v / 0.35
    
    # Precompute trigonometric functions
    sin_t = np.sin(t)
    cos_t = np.cos(t)
    
    # Construct basis terms
    # 0
    b0 = 1.0
    # 1
    b1 = X
    # 2
    b2 = V
    # 3
    b3 = X**3
    # 4
    b4 = X**2 * V
    # 5
    b5 = sin_t
    # 6
    b6 = cos_t
    # 7
    b7 = X * sin_t
    # 8
    b8 = X * cos_t
    # 9
    b9 = V * cos_t
    
    # Combine with parameters
    y = (params[0] * b0 + 
         params[1] * b1 + 
         params[2] * b2 + 
         params[3] * b3 + 
         params[4] * b4 + 
         params[5] * b5 + 
         params[6] * b6 + 
         params[7] * b7 + 
         params[8] * b8 + 
         params[9] * b9)
         
    return y
# EVOLVE-BLOCK-END
