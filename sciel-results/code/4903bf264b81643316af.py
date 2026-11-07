"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a damped nonlinear oscillator with time-dependent driving force.
    Uses a physically-motivated basis with normalized inputs to aid BFGS convergence.
    """
    # Normalize inputs to O(1) scale for better conditioning
    # t range [30, 50] -> center 40, scale 10 -> [-1, 1]
    t_norm = (t - 40.0) / 10.0
    
    # x range approx [-0.21, 0.18] -> scale 0.21
    x_norm = x / 0.21
    
    # v range approx [-0.34, 0.36] -> scale 0.36
    v_norm = v / 0.36
    
    # Extract parameters
    p0 = params[0]  # Bias / Constant
    p1 = params[1]  # Linear restoring (x)
    p2 = params[2]  # Nonlinear restoring (x^3)
    p3 = params[3]  # Linear damping (v)
    p4 = params[4]  # Nonlinear damping (x^2 * v)
    p5 = params[5]  # Quadratic term (x^2)
    p6 = params[6]  # Cosine amplitude
    p7 = params[7]  # Frequency multiplier
    p8 = params[8]  # Phase
    p9 = params[9]  # Sine amplitude
    
    # Calculate algebraic terms
    term_x = x_norm
    term_x3 = x_norm**3
    term_v = v_norm
    term_x2v = x_norm**2 * v_norm
    term_x2 = x_norm**2
    
    # Calculate drive terms
    # Scale frequency to start in a more plausible range (approx 5 rad/norm_unit)
    freq = p7 * 5.0
    angle = freq * t_norm + p8
    term_cos = np.cos(angle)
    term_sin = np.sin(angle)
    
    # Combine
    result = (p0 * 1.0 + 
              p1 * term_x + 
              p2 * term_x3 + 
              p3 * term_v + 
              p4 * term_x2v + 
              p5 * term_x2 + 
              p6 * term_cos + 
              p9 * term_sin)
              
    return result
# EVOLVE-BLOCK-END
