"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Flexible symbolic regression basis with 10 parameters:
    - Polynomial up to degree 5 (6 terms)
    - Sin with learnable frequency and amplitude (2 params)
    - Cos with learnable frequency and amplitude (2 params)
    
    params layout (10 elements):
      params[0]:  a0 (constant)
      params[1]:  a1 (x coeff)
      params[2]:  a2 (x^2 coeff)
      params[3]:  a3 (x^3 coeff)
      params[4]:  a4 (x^4 coeff)
      params[5]:  a5 (x^5 coeff)
      params[6]:  b1 (sin amplitude)
      params[7]:  f1 (sin frequency)
      params[8]:  b2 (cos amplitude)
      params[9]:  f2 (cos frequency)
    """
    a0 = params[0]
    a1 = params[1]
    a2 = params[2]
    a3 = params[3]
    a4 = params[4]
    a5 = params[5]
    b1 = params[6]
    f1 = params[7]
    b2 = params[8]
    f2 = params[9]
    
    # Polynomial part
    poly = a0 + a1 * x + a2 * x**2 + a3 * x**3 + a4 * x**4 + a5 * x**5
    
    # Trigonometric part - clip frequencies to avoid numerical issues
    f1_clipped = np.clip(f1, -20, 20)
    f2_clipped = np.clip(f2, -20, 20)
    trig = b1 * np.sin(f1_clipped * x) + b2 * np.cos(f2_clipped * x)
    
    return poly + trig
# EVOLVE-BLOCK-END
