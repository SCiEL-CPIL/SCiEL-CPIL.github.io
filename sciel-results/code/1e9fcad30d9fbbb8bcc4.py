"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 115.0  # Safety margin for 120s limit

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Deadline guard
    if time.time() - _START_TIME > _DEADLINE:
        return np.zeros_like(x)

    # Structural Change:
    # 1. Scale polynomial inputs to [-1, 1] to improve conditioning.
    # 2. Replace tanh(x) with exp(-x^2) to better capture localized peaks.
    # 3. Keep trigonometric terms on raw x to capture natural frequencies.
    
    z = x / 3.0
    
    # Polynomial basis on scaled x
    b0 = np.ones_like(x)
    b1 = z
    b2 = z ** 2
    b3 = z ** 3
    b4 = z ** 4
    
    # Trigonometric basis on raw x
    b5 = np.sin(x)
    b6 = np.cos(x)
    b7 = np.sin(2.0 * x)
    b8 = np.cos(2.0 * x)
    
    # Localized Gaussian basis
    b9 = np.exp(-x ** 2)
    
    y = (params[0] * b0 + 
         params[1] * b1 + 
         params[2] * b2 + 
         params[3] * b3 + 
         params[4] * b4 + 
         params[5] * b5 + 
         params[6] * b6 + 
         params[7] * b7 + 
         params[8] * b8 + 
         params[9] * b9)
         
    return y
# EVOLVE-BLOCK-END
