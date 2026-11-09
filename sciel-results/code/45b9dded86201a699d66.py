import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator acceleration using a bounded feature basis.
    
    The model is linear in parameters, with features capturing:
    - Position: linear, tanh-bounded cubic, quadratic
    - Velocity: linear, tanh-bounded cubic, quadratic  
    - Time: cos(t), sin(t) for periodic driving
    - Cross: x*v coupling
    - Constant offset
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Position basis (spring force components)
    f_x1 = x
    f_x2 = x * np.tanh(x)
    f_x3 = x * x
    
    # Velocity basis (damping force components)
    f_v1 = v
    f_v2 = v * np.tanh(v)
    f_v3 = v * v
    
    # Time basis (periodic driving force)
    f_t1 = np.cos(t)
    f_t2 = np.sin(t)
    
    # Cross-coupling
    f_xv = x * v
    
    # Linear combination of features
    return (p0 * f_x1 + p1 * f_x2 + p2 * f_x3 +
            p3 * f_v1 + p4 * f_v2 + p5 * f_v3 +
            p6 * f_t1 + p7 * f_t2 +
            p8 * f_xv +
            p9)
