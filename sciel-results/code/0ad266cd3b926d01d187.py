import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Basis functions for a generalized nonlinear oscillator
    # Assuming drive frequency is 1.0 rad/s
    
    # Drive terms (fundamental frequency)
    c1 = np.cos(t)
    s1 = np.sin(t)
    
    # Velocity terms (Damping)
    v1 = v
    v3 = v**3
    
    # Position terms (Restoring force)
    x1 = x
    x3 = x**3
    x2 = x**2
    
    # Interaction terms
    xv = x * v
    x2v = x**2 * v  # Van der Pol term
    
    # Bias
    ones = np.ones_like(t)
    
    # Combine with fitted parameters
    # Order:
    # 0: cos(t)
    # 1: sin(t)
    # 2: v
    # 3: v^3
    # 4: x
    # 5: x^3
    # 6: x^2 v
    # 7: x v
    # 8: x^2
    # 9: 1
    
    return (params[0] * c1 + 
            params[1] * s1 + 
            params[2] * v1 + 
            params[3] * v3 + 
            params[4] * x1 + 
            params[5] * x3 + 
            params[6] * x2v + 
            params[7] * xv + 
            params[8] * x2 + 
            params[9] * ones)
