# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form acceleration of a nonlinear damped oscillator with Van der Pol structure.
    Uses 10 fitted constants in a basis that explicitly captures amplitude-dependent
    damping: energy injection at small amplitudes, dissipation at large amplitudes.
    
    Van der Pol physics: ddot{x} = mu*(1 - x^2)*v - omega^2*x
    This motivates terms: v, x^2*v (nonlinear damping), x (restoring), x^3 (softening)
    
    Structure:
    - p0 * x        : linear restoring
    - p1 * v        : linear damping
    - p2 * x^2 * v  : Van der Pol nonlinear damping (key term)
    - p3 * x^3      : cubic restoring (softening/hardening)
    - p4 * x^4 * v  : higher-order damping correction
    - p5 * x * v^2  : velocity nonlinearity
    - p6 * x^2      : quadratic restoring (asymmetry)
    - p7 * v^2      : quadratic velocity term
    - p8 * x^5      : quintic restoring
    - p9 * x^3 * v  : cubic restoring cross with velocity
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    x5 = x4 * x
    v2 = v * v
    x2v = x2 * v
    x4v = x4 * v
    xv2 = x * v2
    x3v = x3 * v
    
    return (p0 * x + 
            p1 * v + 
            p2 * x2v + 
            p3 * x3 + 
            p4 * x4v + 
            p5 * xv2 + 
            p6 * x2 + 
            p7 * v2 + 
            p8 * x5 + 
            p9 * x3v)
# EVOLVE-BLOCK-END
