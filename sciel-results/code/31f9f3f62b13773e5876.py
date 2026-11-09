"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a damped nonlinear oscillator with driving force.
    
    Uses a polynomial basis in scaled x and v to capture nonlinear dynamics.
    Scaling inputs to unit range improves condition number for BFGS fitting.
    
    params: 10 fitted constants
    """
    # Scale inputs to approximately unit range based on training ranges
    # x range: [-0.5728, 0.5275], center ~ -0.0227, half-range ~ 0.5502
    # v range: [-0.1754, 0.2821], center ~ 0.0534, half-range ~ 0.2288
    x_scale = 1.0 / 0.5502
    v_scale = 1.0 / 0.2288
    x_center = -0.0227
    v_center = 0.0534
    
    xs = (x - x_center) * x_scale
    vs = (v - v_center) * v_scale
    
    # Polynomial basis terms (up to degree 3)
    # 10 terms total
    p = params
    
    # Linear terms
    t1 = xs
    t2 = vs
    
    # Quadratic terms
    t3 = xs * xs
    t4 = vs * vs
    t5 = xs * vs
    
    # Cubic terms
    t6 = xs * xs * xs
    t7 = vs * vs * vs
    t8 = xs * xs * vs
    t9 = xs * vs * vs
    
    # Constant / bias
    t10 = np.ones_like(xs)
    
    return (p[0] * t1 + 
            p[1] * t2 + 
            p[2] * t3 + 
            p[3] * t4 + 
            p[4] * t5 + 
            p[5] * t6 + 
            p[6] * t7 + 
            p[7] * t8 + 
            p[8] * t9 + 
            p[9] * t10)
# EVOLVE-BLOCK-END
