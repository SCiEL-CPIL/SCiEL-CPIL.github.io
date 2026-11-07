import numpy as np
import time

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Physics-constrained rational function model for a nonlinear oscillator.
    
    The acceleration a is modeled as a ratio of two polynomials:
        a = N(x, v, t) / D(x, v, t)
    
    This structure naturally captures saturation and softening nonlinearities.
    
    Numerator N (degree up to 2 in x, v, and trig terms):
        p0 + p1*x + p2*x^2 + p3*v + p4*x*v + p5*sin(t) + p6*cos(t)
    
    Denominator D (degree up to 1 in x, v):
        1 + p7*x + p8*v + p9*x^2
    
    The denominator ensures the model can capture softening (D > 1 for large |x|)
    or hardening (D < 1 for large |x|) behavior, which additive polynomials cannot.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Precompute terms
    x2 = x * x
    xv = x * v
    sin_t = np.sin(t)
    cos_t = np.cos(t)
    
    # Numerator: quadratic in x, linear in v, with trig driving
    numerator = (p0 + 
                 p1 * x + 
                 p2 * x2 + 
                 p3 * v + 
                 p4 * xv + 
                 p5 * sin_t + 
                 p6 * cos_t)
    
    # Denominator: ensures non-zero and captures softening/hardening
    # Add a small constant to avoid division by zero issues during optimization
    denominator = 1.0 + p7 * x + p8 * v + p9 * x2
    
    # Avoid division by zero by clipping denominator away from zero
    # Use a safe minimum magnitude to prevent numerical instability
    denom_safe = np.where(np.abs(denominator) < 1e-6, 
                          np.sign(denominator + 1e-10) * 1e-6, 
                          denominator)
    
    return numerator / denom_safe
