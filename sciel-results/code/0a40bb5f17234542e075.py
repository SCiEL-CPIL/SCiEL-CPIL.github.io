import numpy as np
import time

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator using a rational (Padé-like) form.
    The denominator provides inherent saturation and better conditioning.
    """
    # Scale inputs to roughly [-1, 1] range to help BFGS
    X = x / 0.6
    V = v / 0.3
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Numerator: complete quadratic polynomial in X and V (5 terms)
    num = (p0 + 
           p1 * X + 
           p2 * V + 
           p3 * X**2 + 
           p4 * V**2)
    
    # Denominator: 1 + cubic terms to provide saturation (5 terms)
    den = 1.0 + (p5 * X**3 + 
                 p6 * V**3 + 
                 p7 * X * V**2 + 
                 p8 * X**2 * V + 
                 p9 * X * V)
    
    # Ensure denominator doesn't get too close to zero for numerical stability
    den_safe = np.where(np.abs(den) < 1e-6, 1e-6 * np.sign(den), den)
    
    return num / den_safe
