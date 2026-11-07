import numpy as np
import time

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Physics-informed basis for a nonlinear oscillator (Duffing/Van der Pol style).
    
    The model is structured as:
        a = -k*x - alpha*x^3 - beta*v - gamma*(1-x^2)*v + ...
    
    We use 10 parameters to capture:
    - Linear restoring: -p0 * x
    - Cubic restoring: -p1 * x^3
    - Linear damping: -p2 * v
    - Van der Pol damping: -p3 * (1 - x^2) * v
    - Additional cross terms and higher-order corrections:
      p4 * x*v, p5 * x^2, p6 * v^2, p7 * x^2*v, p8 * x*v^2, p9 * constant
    
    Scaling is applied to improve numerical conditioning.
    """
    # Scaling factors based on training ranges
    x_scale = 0.6
    v_scale = 0.3
    
    xs = x / x_scale
    vs = v / v_scale
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Core physics terms
    linear_restoring = -p0 * xs
    cubic_restoring = -p1 * xs**3
    linear_damping = -p2 * vs
    vdp_damping = -p3 * (1.0 - xs**2) * vs
    
    # Additional terms for flexibility
    cross_xv = p4 * xs * vs
    quad_x = p5 * xs**2
    quad_v = p6 * vs**2
    x2v = p7 * xs**2 * vs
    xv2 = p8 * xs * vs**2
    const = p9
    
    return linear_restoring + cubic_restoring + linear_damping + vdp_damping + \
           cross_xv + quad_x + quad_v + x2v + xv2 + const
