import numpy as np
import time

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Polynomial interaction model with bounded nonlinearities.
    params: [c0, c1, c2, c3, c4, c5, c6, c7, c8, c9]
    Uses normalized time and bounded nonlinear terms (tanh) for better BFGS conditioning.
    """
    c0 = params[0]
    c1 = params[1]
    c2 = params[2]
    c3 = params[3]
    c4 = params[4]
    c5 = params[5]
    c6 = params[6]
    c7 = params[7]
    c8 = params[8]
    c9 = params[9]
    
    # Normalize time to approx [-1, 1] for better conditioning
    t_norm = (t - 40.0) / 10.0
    
    # Bounded nonlinear features to avoid ill-conditioning
    x_tanh = np.tanh(x)
    v_tanh = np.tanh(v)
    x2 = x * x
    v2 = v * v
    x_v = x * v
    t_x = t_norm * x
    t_v = t_norm * v
    t2 = t_norm * t_norm
    
    # Linear in params: smooth, well-conditioned basis
    result = (c0 
              + c1 * x + c2 * v 
              + c3 * x2 + c4 * v2 + c5 * x_v
              + c6 * x_tanh + c7 * v_tanh
              + c8 * (t_x + t_v) 
              + c9 * t2)
    
    return result
