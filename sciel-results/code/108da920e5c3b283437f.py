# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator acceleration with parametric excitation and rational saturation.
    
    Model: a = (p0*x + p1*v + p2*x^3 + p3*v^3 + p4*cos(w*tc)*x + p5*sin(w*tc) 
              + p6*x^2 + p7*v^2 + p8*x*v + p9) / (1 + eps*x^2)
    
    Key structural features:
    - Parametric excitation (cos(w*tc)*x) - standard for nonlinear oscillators
    - Rational denominator provides saturation/softening nonlinearity
    - Time centered at midpoint of [30, 50] for numerical stability
    - Cubic stiffness and damping terms (Duffing-type)
    - Quadratic correction terms
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Center time for numerical stability (midpoint of [30, 50])
    tc = t - 40.0
    
    # Fixed frequency for parametric excitation
    w = 1.0
    
    # Precompute powers
    x2 = x * x
    x3 = x2 * x
    v2 = v * v
    v3 = v2 * v
    
    # Parametric excitation terms
    cos_drive = np.cos(w * tc)
    sin_drive = np.sin(w * tc)
    
    # Numerator: Duffing + parametric forcing + corrections
    numerator = (
        p0 * x +
        p1 * v +
        p2 * x3 +
        p3 * v3 +
        p4 * cos_drive * x +
        p5 * sin_drive +
        p6 * x2 +
        p7 * v2 +
        p8 * x * v +
        p9
    )
    
    # Denominator: rational saturation term
    # Small fixed epsilon ensures denominator is bounded away from zero
    eps = 0.01
    denominator = 1.0 + eps * x2
    
    result = numerator / denominator
    
    return result
# EVOLVE-BLOCK-END
