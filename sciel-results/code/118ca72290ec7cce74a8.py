import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression for a nonlinear oscillator using a rational function basis.
    
    The function is defined as:
    f(x, v) = (p0 + p1*x + p2*v + p3*x^2 + p4*v^2 + p5*x*v) / 
              (1 + p6*x + p7*v + p8*x^2 + p9*v^2)
    
    This allows the parameters to modulate the response non-linearly, 
    capturing stiffness changes or damping effects that depend on position/velocity.
    """
    p = params
    
    # Numerator terms
    num = (p[0] + 
           p[1] * x + 
           p[2] * v + 
           p[3] * x**2 + 
           p[4] * v**2 + 
           p[5] * x * v)
           
    # Denominator terms
    # Adding 1 ensures the denominator is non-zero for small x, v 
    # even if parameters are initialized to 1.
    den = (1.0 + 
           p[6] * x + 
           p[7] * v + 
           p[8] * x**2 + 
           p[9] * v**2)
           
    # Avoid division by zero (though unlikely given the ranges and '1+' term)
    # np.where or np.clip can be used, but simple division is fast.
    # To be safe against numerical issues if den gets close to 0:
    den_safe = np.where(np.abs(den) < 1e-8, 1e-8, den)
    
    return num / den_safe
