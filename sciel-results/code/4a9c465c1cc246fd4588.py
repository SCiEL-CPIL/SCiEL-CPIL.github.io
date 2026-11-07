"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator acceleration using a centered polynomial basis.
    Centers t around 40 to improve numerical conditioning for the optimizer.
    Includes linear, quadratic, and cross terms.
    """
    c0, c1, c2, c3, c4, c5, c6, c7, c8, c9 = params
    
    # Center variables to reduce magnitude and improve optimization stability
    t_c = t - 40.0
    x_c = x
    v_c = v
    
    # Basis functions
    # 1. Constant
    b0 = np.ones_like(x_c)
    # 2. Linear terms
    b1 = x_c
    b2 = v_c
    b3 = t_c
    # 3. Quadratic terms
    b4 = x_c * x_c
    b5 = v_c * v_c
    b6 = t_c * t_c
    # 4. Cross terms
    b7 = x_c * v_c
    b8 = x_c * t_c
    b9 = v_c * t_c
    
    # Linear combination of basis functions
    result = (c0 * b0 
              + c1 * b1 
              + c2 * b2 
              + c3 * b3 
              + c4 * b4 
              + c5 * b5 
              + c6 * b6 
              + c7 * b7 
              + c8 * b8 
              + c9 * b9)
    return result
# EVOLVE-BLOCK-END
