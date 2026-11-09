"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression for a nonlinear oscillator acceleration.
    
    Uses a Fourier-based ansatz combined with polynomial terms to capture
    periodic and nonlinear dynamics. The oscillator equation is modeled as:
    
    a = c0 + c1*x + c2*v + c3*x^2 + c4*v^2 + c5*x*v 
      + c6*sin(k*x) + c7*sin(k*v) + c8*cos(k*x) + c9*cos(k*v)
    
    where k is a fixed frequency scale (set to 2*pi to match typical oscillator periods).
    This structure allows BFGS to fit amplitude, phase, and nonlinear coupling terms
    directly, leveraging the periodic nature of oscillatory systems.
    
    Parameters:
        x: Position (array)
        v: Velocity (array)
        params: Array of 10 fitted constants
            params[0]: Constant offset (c0)
            params[1]: Linear position coefficient (c1)
            params[2]: Linear velocity coefficient (c2)
            params[3]: Quadratic position coefficient (c3)
            params[4]: Quadratic velocity coefficient (c4)
            params[5]: Cross term coefficient (c5)
            params[6]: Sin(k*x) amplitude (c6)
            params[7]: Sin(k*v) amplitude (c7)
            params[8]: Cos(k*x) amplitude (c8)
            params[9]: Cos(k*v) amplitude (c9)
    
    Returns:
        Acceleration prediction (array, same shape as x)
    """
    k = 2.0 * np.pi  # Fixed frequency scale for Fourier terms
    
    c0 = params[0]
    c1 = params[1]
    c2 = params[2]
    c3 = params[3]
    c4 = params[4]
    c5 = params[5]
    c6 = params[6]
    c7 = params[7]
    c8 = params[8]
    c9 = params[9]
    
    # Polynomial terms
    poly = c0 + c1 * x + c2 * v + c3 * x**2 + c4 * v**2 + c5 * x * v
    
    # Fourier terms
    fourier = c6 * np.sin(k * x) + c7 * np.sin(k * v) + c8 * np.cos(k * x) + c9 * np.cos(k * v)
    
    return poly + fourier
# EVOLVE-BLOCK-END
