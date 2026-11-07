import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator acceleration using a physics-informed basis of smooth,
    bounded nonlinearities (sin, tanh) to capture periodic restoring forces and
    saturation while avoiding ill-conditioning of high-degree monomials.
    
    Basis functions:
    - sin(x), x, tanh(x): restoring force components
    - sin(v), v, tanh(v): damping components  
    - sin(x)*v, x*sin(v): cross terms
    - sin(x)^2, tanh(x)*sin(v): nonlinear coupling
    - constant
    """
    p = params
    
    # Compute trigonometric and hyperbolic basis functions
    sin_x = np.sin(x)
    tanh_x = np.tanh(x)
    sin_v = np.sin(v)
    tanh_v = np.tanh(v)
    
    # Restore force terms
    f0 = sin_x
    f1 = x
    f2 = tanh_x
    
    # Damping terms
    f3 = sin_v
    f4 = v
    f5 = tanh_v
    
    # Cross terms
    f6 = sin_x * v
    f7 = x * sin_v
    
    # Nonlinear coupling
    f8 = sin_x * sin_x
    f9 = tanh_x * sin_v
    
    return p[0]*f0 + p[1]*f1 + p[2]*f2 + p[3]*f3 + p[4]*f4 + p[5]*f5 + p[6]*f6 + p[7]*f7 + p[8]*f8 + p[9]*f9
