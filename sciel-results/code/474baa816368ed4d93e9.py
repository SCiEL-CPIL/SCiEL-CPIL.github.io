"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Equation of motion for a nonlinear oscillator with time-dependent driving force.
    
    Model:
    a = c1*x + c2*v + c3*x^3 + c4*x^2*v + c5*v^3 + c6*sin(w*t + p) + c7*cos(w*t + p) + c8
    
    Parameters:
    params[0]: c1 (linear spring)
    params[1]: c2 (linear damping)
    params[2]: c3 (cubic stiffness, Duffing)
    params[3]: c4 (nonlinear damping, Van der Pol style x^2*v)
    params[4]: c5 (cubic damping v^3)
    params[5]: c6 (drive sin amplitude)
    params[6]: c7 (drive cos amplitude)
    params[7]: w  (drive frequency)
    params[8]: p  (drive phase)
    params[9]: c8 (constant offset)
    """
    c1, c2, c3, c4, c5, c6, c7, w, p, c8 = params
    
    # State terms
    term_x = c1 * x
    term_v = c2 * v
    term_x3 = c3 * (x ** 3)
    term_x2v = c4 * (x ** 2) * v
    term_v3 = c5 * (v ** 3)
    
    # Drive terms
    theta = w * t + p
    term_drive = c6 * np.sin(theta) + c7 * np.cos(theta)
    
    # Constant
    term_const = c8
    
    return term_x + term_v + term_x3 + term_x2v + term_v3 + term_drive + term_const
# EVOLVE-BLOCK-END
