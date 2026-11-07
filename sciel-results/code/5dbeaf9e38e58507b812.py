"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear damped oscillator with time-dependent driving.
    Model: a = c0*x + c1*x^3 + c2*v + c3*v^3 + c4*sin(c5*t_c) + c6*cos(c5*t_c) + c7*x*v + c8*x^2 + c9
    """
    c0, c1, c2, c3, c4, c5, c6, c7, c8, c9 = params
    
    # Center time to improve conditioning for low frequencies
    t_c = t - 40.0
    
    # Calculate terms
    # Restoring forces (Duffing)
    term_x = c0 * x
    term_x3 = c1 * x**3
    
    # Damping forces (Van der Pol)
    term_v = c2 * v
    term_v3 = c3 * v**3
    
    # Driving force (Sinusoidal with fitted frequency)
    # Using centered time
    drive_sin = c4 * np.sin(c5 * t_c)
    drive_cos = c6 * np.cos(c5 * t_c)
    
    # Interaction and quadratic terms
    term_xv = c7 * x * v
    term_x2 = c8 * x**2
    
    # Constant offset
    const = c9
    
    result = term_x + term_x3 + term_v + term_v3 + drive_sin + drive_cos + term_xv + term_x2 + const
    
    # Safety check for non-finite values
    if not np.all(np.isfinite(result)):
        return np.zeros_like(result)
        
    return result
# EVOLVE-BLOCK-END
