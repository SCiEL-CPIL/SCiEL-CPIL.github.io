import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers a function using a parametric trigonometric Fourier series basis.
    
    The model is linear in the 10 parameters:
    f(x, y) = p0*1 + p1*sin(pi*x) + p2*cos(pi*x) + p3*sin(pi*y) + p4*cos(pi*y) +
              p5*sin(pi*x)*cos(pi*y) + p6*cos(pi*x)*sin(pi*y) +
              p7*sin(2*pi*x) + p8*sin(2*pi*y) + p9*cos(2*pi*x)*cos(2*pi*y)
    
    This basis captures periodic behaviors and interactions that polynomial bases might miss.
    """
    # Ensure inputs are 1D arrays
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    
    # Compute basis functions
    # Use pi to map [-1, 1] to [-pi, pi] for standard trig functions
    # sin(pi * [-1, 1]) -> sin([-pi, pi]) which is a full half-period? No, sin(-pi)=0, sin(pi)=0.
    # Actually, mapping [-1,1] to [-pi, pi] is good for frequency 1.
    
    pi = np.pi
    
    sin_px = np.sin(pi * x)
    cos_px = np.cos(pi * x)
    sin_py = np.sin(pi * y)
    cos_py = np.cos(pi * y)
    
    sin_2px = np.sin(2.0 * pi * x)
    sin_2py = np.sin(2.0 * pi * y)
    cos_2px = np.cos(2.0 * pi * x)
    cos_2py = np.cos(2.0 * pi * y)
    
    # Construct the 10 basis functions
    phi0 = np.ones_like(x)
    phi1 = sin_px
    phi2 = cos_px
    phi3 = sin_py
    phi4 = cos_py
    phi5 = sin_px * cos_py
    phi6 = cos_px * sin_py
    phi7 = sin_2px
    phi8 = sin_2py
    phi9 = cos_2px * cos_2py
    
    # Stack features into a matrix (N, 10)
    Phi = np.stack((phi0, phi1, phi2, phi3, phi4, phi5, phi6, phi7, phi8, phi9), axis=1)
    
    # Compute predictions: Phi (N, 10) @ params (10,) -> (N,)
    pred = Phi @ params
    
    # Ensure finite values
    if not np.all(np.isfinite(pred)):
        # Fallback: return zeros or simple linear if nan/inf
        # This should rarely happen with trig functions on bounded domain
        return np.zeros_like(x)
        
    return pred
