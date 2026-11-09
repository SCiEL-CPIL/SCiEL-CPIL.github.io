import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Basis functions for a driven nonlinear oscillator (e.g., Duffing)
    # Driving force terms (fundamental and 2nd harmonic)
    c1 = np.cos(t)
    s1 = np.sin(t)
    c2 = np.cos(2 * t)
    s2 = np.sin(2 * t)
    
    # State terms (restoring force and damping)
    x1 = x
    x3 = x**3
    v1 = v
    xv = x * v
    v2 = v**2
    
    # Constant term
    ones = np.ones_like(t)
    
    # Combine with fitted parameters
    # Order: c1, s1, c2, s2, x1, x3, v1, xv, v2, ones
    return (params[0] * c1 + 
            params[1] * s1 + 
            params[2] * c2 + 
            params[3] * s2 + 
            params[4] * x1 + 
            params[5] * x3 + 
            params[6] * v1 + 
            params[7] * xv + 
            params[8] * v2 + 
            params[9] * ones)
