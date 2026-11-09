import numpy as np
import time

_DEADLINE = time.time() + 135.0

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression for a nonlinear oscillator: a = f(x, v, t)
    
    Combines the multiplicative/physical structure of the best inspiration (score 57.8)
    with refined basis terms from other inspirations. Uses normalized inputs to help
    BFGS converge from the all-ones initial guess.
    
    Parameters (10 total):
    p0: bias/offset
    p1: linear stiffness (x)
    p2: cubic stiffness (x^3) - Duffing nonlinearity
    p3: linear damping (v)
    p4: nonlinear damping (v*|v|)
    p5: position-velocity coupling (x*v)
    p6: sinusoidal forcing amplitude (sin(t))
    p7: cosinusoidal forcing amplitude (cos(t))
    p8: quadratic position term (x^2)
    p9: quadratic velocity term (v^2)
    """
    # Normalize inputs to improve BFGS conditioning
    # t: [30, 50] -> center 40, scale 10
    # x: [-0.2056, 0.1749] -> center -0.015, scale 0.2
    # v: [-0.3391, 0.3624] -> center 0.01, scale 0.35
    t_n = (t - 40.0) / 10.0
    x_n = (x + 0.015) / 0.2
    v_n = (v - 0.01) / 0.35
    
    # Extract parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Compute basis functions
    x2 = x_n * x_n
    x3 = x2 * x_n
    v_abs = np.abs(v_n)
    v_nonlinear = v_n * v_abs  # quadratic damping
    v2 = v_n * v_n
    xv = x_n * v_n
    St = np.sin(t)
    Ct = np.cos(t)
    
    # Linear combination of basis functions
    res = (p0 + 
           p1 * x_n + 
           p2 * x3 + 
           p3 * v_n + 
           p4 * v_nonlinear + 
           p5 * xv + 
           p6 * St + 
           p7 * Ct + 
           p8 * x2 + 
           p9 * v2)
    
    return res
