"""Nonlinear oscillator symbolic regression with Chebyshev-basis acceleration."""
import numpy as np
import time


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration model for a nonlinear oscillator.
    
    Uses a combination of:
    - Time-dependent sinusoidal driving force
    - Nonlinear restoring force (Duffing-type)
    - Linear and nonlinear damping
    - Cross-coupling terms
    
    params: array of 10 fitted constants
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Time-dependent driving force: sinusoidal with amplitude, frequency, phase
    drive = p0 * np.sin(p1 * t + p2)
    
    # Nonlinear restoring force: linear + cubic (Duffing) + quadratic
    restoring = p3 * x + p4 * x**3 + p8 * x**2
    
    # Damping: linear + cubic (nonlinear drag)
    damping = p5 * v + p6 * v**3
    
    # Cross-coupling term
    coupling = p7 * x * v
    
    # Constant bias
    bias = p9
    
    return drive + restoring + damping + coupling + bias
