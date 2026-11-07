import numpy as np

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Rational function model: ratio of two linear-in-features polynomials.
    
    Numerator:  p0 + p1*s + p2*t + p3*ph + p4*b + p5*s*t + p6*s*ph
    Denominator: 1 + p7*s + p8*t + p9*ph
    
    This structure:
    - Captures interactions between substrate and other factors
    - Built-in saturation via denominator
    - Smooth and differentiable everywhere (denominator >= 1 for positive params)
    - Only 10 parameters total
    """
    p = params
    
    # Use raw inputs but scale for numerical stability
    # b: [1, 5], s: [1, 5], temp: [1, 40], pH: [2, 12]
    # Scale to roughly [0, 1] for better conditioning
    b_s = b / 5.0
    s_s = s / 5.0
    t_s = temp / 40.0
    ph_s = pH / 12.0
    
    # Numerator: linear + key interaction terms
    num = (p[0] 
           + p[1] * s_s 
           + p[2] * t_s 
           + p[3] * ph_s 
           + p[4] * b_s 
           + p[5] * s_s * t_s 
           + p[6] * s_s * ph_s)
    
    # Denominator: ensures saturation behavior
    # Add 1.0 to ensure denominator is positive and avoids division by zero
    den = 1.0 + p[7] * s_s + p[8] * t_s + p[9] * ph_s
    
    # Ensure denominator is well-behaved (add small epsilon for safety)
    den = np.where(np.abs(den) < 1e-6, 1e-6, den)
    
    return num / den
