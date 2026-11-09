"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator model with normalized inputs for better conditioning.
    Structure:
    y = p0*xn + p1*xn^3 + p2*vn + p3*vn^3 
      + p4*sin(t) + p5*cos(t) + p6*sin(2t) + p7*cos(2t) 
      + p8*t + p9
    where xn = x/0.2056, vn = v/0.3391
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Normalize inputs to O(1) range to help BFGS convergence
    x_norm = x / 0.2056
    v_norm = v / 0.3391
    
    # Nonlinear restoring and damping forces
    term_x = p0 * x_norm + p1 * (x_norm ** 3)
    term_v = p2 * v_norm + p3 * (v_norm ** 3)
    
    # Driving force (assumed periodic with fundamental frequency 1 and 2nd harmonic)
    term_drive = p4 * np.sin(t) + p5 * np.cos(t) + p6 * np.sin(2.0 * t) + p7 * np.cos(2.0 * t)
    
    # Linear trend and offset
    term_t = p8 * t + p9
    
    return term_x + term_v + term_drive + term_t
# EVOLVE-BLOCK-END
