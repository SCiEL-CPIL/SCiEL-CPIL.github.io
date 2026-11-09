"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Map x from [-3, 3] to t in [-1, 1] for better conditioning
    t = x / 3.0
    
    # Chebyshev polynomials of the first kind T_n(t)
    # T_0(t) = 1
    # T_1(t) = t
    # T_{n+1}(t) = 2t T_n(t) - T_{n-1}(t)
    
    T0 = np.ones_like(t)
    T1 = t
    T2 = 2 * t * T1 - T0
    T3 = 2 * t * T2 - T1
    T4 = 2 * t * T3 - T2
    T5 = 2 * t * T4 - T3
    T6 = 2 * t * T5 - T4
    
    # Additional basis functions for non-polynomial behavior
    # Sinusoidal components with period 6 (since t is in [-1,1], sin(pi*t) goes 0->1->0->-1->0)
    S1 = np.sin(np.pi * t)
    C1 = np.cos(np.pi * t)
    
    # Sigmoid-like component
    H1 = np.tanh(t)
    
    # Combine with parameters
    # params[0] ... params[9]
    y = (params[0] * T0 + 
         params[1] * T1 + 
         params[2] * T2 + 
         params[3] * T3 + 
         params[4] * T4 + 
         params[5] * T5 + 
         params[6] * T6 + 
         params[7] * S1 + 
         params[8] * C1 + 
         params[9] * H1)
         
    return y
# EVOLVE-BLOCK-END
