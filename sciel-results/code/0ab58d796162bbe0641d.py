import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Driven Duffing oscillator: acceleration = drive + restoring + damping
    # Drive: fundamental and 2nd harmonic
    c1 = np.cos(t)
    s1 = np.sin(t)
    c2 = np.cos(2.0 * t)
    s2 = np.sin(2.0 * t)
    
    # Restoring force: linear, quadratic, cubic (Duffing)
    x1 = x
    x2 = x * x
    x3 = x * x * x
    
    # Damping: linear in v, and x*v interaction
    v1 = v
    xv = x * v
    
    # Constant (bias)
    ones = np.ones_like(t)
    
    # Combine: 10 fitted parameters
    return (params[0] * c1 +
            params[1] * s1 +
            params[2] * c2 +
            params[3] * s2 +
            params[4] * x1 +
            params[5] * x2 +
            params[6] * x3 +
            params[7] * v1 +
            params[8] * xv +
            params[9] * ones)
