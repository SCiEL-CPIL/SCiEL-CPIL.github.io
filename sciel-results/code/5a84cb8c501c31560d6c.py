import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator model with normalized inputs.
    
    Model:
    a = p0*X + p1*X^3 + p2*V + p3*V^3 + p4*X^2*V 
      + p5*sin(t) + p6*cos(t) + p7*X*V + p8*X^2 + p9
    
    Where X = x/0.2, V = v/0.3.
    """
    # Normalization
    X = x / 0.2
    V = v / 0.3
    
    # Powers
    X2 = X * X
    X3 = X2 * X
    V2 = V * V
    V3 = V2 * V
    
    # Trig
    St = np.sin(t)
    Ct = np.cos(t)
    
    # Extract params
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Compute
    # Using Horner-like or direct sum for clarity and speed
    res = (p0 * X + 
           p1 * X3 + 
           p2 * V + 
           p3 * V3 + 
           p4 * X2 * V + 
           p5 * St + 
           p6 * Ct + 
           p7 * X * V + 
           p8 * X2 + 
           p9)
           
    return res
