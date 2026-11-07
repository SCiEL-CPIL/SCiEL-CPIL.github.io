"""Symbolic regression: nonlinear oscillator acceleration with Physics-Motivated Polynomial/Trigonometric Basis."""
# EVOLVE-BLOCK-START
import numpy as np
import warnings

warnings.filterwarnings('ignore')

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Physics-motivated basis for a nonlinear oscillator.
    
    The model assumes the acceleration is a linear combination of:
    - Nonlinear restoring forces: x, x^3, x^5
    - Damping terms: v, v^3
    - Coupling terms: x*v, x^2*v
    - Periodic driving terms: cos(t), sin(t), cos(2t)
    
    This covers Duffing, Van der Pol, and driven oscillator dynamics.
    10 parameters total.
    """
    p = params
    
    # Precompute basis functions
    # Restoring forces
    phi_x = x
    phi_x3 = x * x * x
    phi_x5 = phi_x3 * x * x
    
    # Damping
    phi_v = v
    phi_v3 = v * v * v
    
    # Coupling
    phi_xv = x * v
    phi_x2v = x * x * v
    
    # Driving terms (assuming fundamental frequency 1 and 2)
    phi_cos_t = np.cos(t)
    phi_sin_t = np.sin(t)
    phi_cos_2t = np.cos(2.0 * t)
    
    # Combine linearly
    # Order of params:
    # 0: x
    # 1: x^3
    # 2: v
    # 3: x*v
    # 4: v^3
    # 5: x^2*v
    # 6: x^5
    # 7: cos(t)
    # 8: sin(t)
    # 9: cos(2t)
    
    result = (p[0] * phi_x + 
              p[1] * phi_x3 + 
              p[2] * phi_v + 
              p[3] * phi_xv + 
              p[4] * phi_v3 + 
              p[5] * phi_x2v + 
              p[6] * phi_x5 + 
              p[7] * phi_cos_t + 
              p[8] * phi_sin_t + 
              p[9] * phi_cos_2t)
              
    return result
# EVOLVE-BLOCK-END
