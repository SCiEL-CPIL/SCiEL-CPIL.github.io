"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE_SECONDS = 140.0
_START = time.time()


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Acceleration of a damped nonlinear oscillator with time-dependent driving force.
    
    Uses 10 basis features to span common nonlinear oscillator dynamics:
    constant, t_centered, x, v, x^2, x^3, v^2, x*v, sin(omega*t_c), cos(omega*t_c)
    """
    # params: [c_bias, c_t, c_x, c_v, c_x2, c_x3, c_v2, c_xv, c_sin, c_cos]
    c_bias = params[0]
    c_t = params[1]
    c_x = params[2]
    c_v = params[3]
    c_x2 = params[4]
    c_x3 = params[5]
    c_v2 = params[6]
    c_xv = params[7]
    c_sin = params[8]
    c_cos = params[9]
    
    # Center time to improve numerical conditioning
    # t range is [30, 50], center at 40
    t_c = t - 40.0
    
    # Calculate polynomial terms
    term_bias = c_bias
    term_t = c_t * t_c
    term_x = c_x * x
    term_v = c_v * v
    term_x2 = c_x2 * (x * x)
    term_x3 = c_x3 * (x * x * x)
    term_v2 = c_v2 * (v * v)
    term_xv = c_xv * (x * v)
    
    # Periodic driving force with fixed frequency to avoid ill-conditioning
    # Use a moderate frequency that might match typical oscillator driving
    phase = 1.0 * t_c  # omega = 1.0 fixed for stability
    
    term_sin = c_sin * np.sin(phase)
    term_cos = c_cos * np.cos(phase)
    
    result = (term_bias + term_t + term_x + term_v + 
              term_x2 + term_x3 + term_v2 + term_xv + 
              term_sin + term_cos)
    
    return result
# EVOLVE-BLOCK-END
