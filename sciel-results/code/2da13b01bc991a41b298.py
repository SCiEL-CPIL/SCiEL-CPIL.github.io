"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a driven nonlinear oscillator with normalized inputs for BFGS conditioning.
    
    Structural change: normalize x and v to unit scale to improve BFGS convergence from
    all-ones start. Use a basis that captures essential nonlinear oscillator physics:
    x^3 (Duffing), v|v| (nonlinear damping), xv (parametric), sin/cos (driving), x^2, v^2.
    
    Model structure (with normalized inputs xn = x/0.35, vn = v/0.35):
    a = p0 + p1*sin(t) + p2*cos(t) + p3*xn + p4*xn^3 + p5*vn + p6*vn*|vn| + p7*xn*vn + p8*xn^2 + p9*vn^2
    """
    # Normalize inputs to roughly unit scale for better BFGS conditioning
    # x range: [-0.3391, 0.3624] -> scale by ~0.35
    # v range: [-0.2056, 0.1749] -> scale by ~0.35
    scale = 0.35
    xn = x / scale
    vn = v / scale
    
    sin_t = np.sin(t)
    cos_t = np.cos(t)
    xn2 = xn * xn
    xn3 = xn2 * xn
    vn2 = vn * vn
    vn_abs = np.abs(vn)
    vn_vnabs = vn * vn_abs
    xn_vn = xn * vn
    
    return (params[0] + 
            params[1] * sin_t + 
            params[2] * cos_t + 
            params[3] * xn + 
            params[4] * xn3 + 
            params[5] * vn + 
            params[6] * vn_vnabs + 
            params[7] * xn_vn + 
            params[8] * xn2 + 
            params[9] * vn2)
# EVOLVE-BLOCK-END
