"""Symbolic regression with 1 input(s) - Fourier series basis."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Fourier series basis with 10 terms.
    
    The basis is strictly linear in params, so BFGS converges to the global
    least-squares solution. Fourier series provide a smooth approximation
    that is excellent for periodic or smooth non-periodic functions on a
    bounded interval.
    
    Basis functions:
    0: 1
    1: cos(pi*x/2)
    2: sin(pi*x/2)
    3: cos(pi*x)
    4: sin(pi*x)
    5: cos(3*pi*x/2)
    6: sin(3*pi*x/2)
    7: cos(2*pi*x)
    8: sin(2*pi*x)
    9: cos(5*pi*x/2)
    """
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    n_points = len(x)
    result = np.zeros(n_points, dtype=np.float64)
    
    # Precompute trigonometric components
    # We need frequencies: 0, 0.5*pi, 1*pi, 1.5*pi, 2*pi, 2.5*pi
    # To save computation, we can compute sin/cos for these specific frequencies
    
    # Frequency 0: DC
    if len(params) > 0:
        result += params[0]
        
    # Frequency 0.5 * pi
    # angle = 0.5 * pi * x
    if len(params) > 1:
        angle = 0.5 * np.pi * x
        result += params[1] * np.cos(angle)
    if len(params) > 2:
        result += params[2] * np.sin(angle)
        
    # Frequency 1.0 * pi
    if len(params) > 3:
        angle = np.pi * x
        result += params[3] * np.cos(angle)
    if len(params) > 4:
        result += params[4] * np.sin(angle)
        
    # Frequency 1.5 * pi
    if len(params) > 5:
        angle = 1.5 * np.pi * x
        result += params[5] * np.cos(angle)
    if len(params) > 6:
        result += params[6] * np.sin(angle)
        
    # Frequency 2.0 * pi
    if len(params) > 7:
        angle = 2.0 * np.pi * x
        result += params[7] * np.cos(angle)
    if len(params) > 8:
        result += params[8] * np.sin(angle)
        
    # Frequency 2.5 * pi
    if len(params) > 9:
        angle = 2.5 * np.pi * x
        result += params[9] * np.cos(angle)
        
    return result
# EVOLVE-BLOCK-END
