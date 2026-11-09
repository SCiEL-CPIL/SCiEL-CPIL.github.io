"""Nonlinear oscillator symbolic regression with coupled and nonlinear terms."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator acceleration model with:
    - Linear restoring force (x)
    - Cubic restoring force (x^3)
    - Linear damping (v)
    - Quadratic damping (v^2)
    - State-dependent damping (x*v)
    - Time-dependent driving (t, t^2)
    - Parametric excitation (t*x)
    - Coupled velocity-time (t*v)
    - Constant bias
    
    Total: 10 fitted parameters
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Nonlinear restoring forces
    restoring = p0 * x + p1 * (x ** 3)
    
    # Damping forces (linear, quadratic, state-dependent)
    damping = p2 * v + p3 * (v ** 2) + p4 * (x * v)
    
    # Time-dependent driving and parametric terms
    driving = p5 * t + p6 * (t ** 2) + p7 * (t * x) + p8 * (t * v)
    
    # Constant bias
    bias = p9
    
    return restoring + damping + driving + bias
# EVOLVE-BLOCK-END
