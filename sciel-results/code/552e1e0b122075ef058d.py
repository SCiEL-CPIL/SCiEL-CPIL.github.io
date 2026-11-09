# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a damped nonlinear oscillator using a rational (Padé-like) form.
    
    y = (p0 + p1*x + p2*v + p3*x^2 + p4*v^2 + p5*x*v + p6*x^3 + p7*v^3)
        / (1 + p8*x^2 + p9*v^2)
    
    This structure captures smooth, bounded nonlinear dynamics. The denominator
    provides inherent saturation and better conditioning for BFGS by preventing
    unbounded growth of higher-order terms.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Numerator terms
    num = (p0 
           + p1 * x 
           + p2 * v 
           + p3 * x**2 
           + p4 * v**2 
           + p5 * x * v 
           + p6 * x**3 
           + p7 * v**3)
    
    # Denominator terms (1 + quadratic corrections)
    den = 1.0 + p8 * x**2 + p9 * v**2
    
    # Ensure denominator is not zero (add small epsilon for numerical stability)
    den = np.where(np.abs(den) < 1e-12, np.sign(den) * 1e-12 + 1e-12, den)
    
    return num / den
# EVOLVE-BLOCK-END
