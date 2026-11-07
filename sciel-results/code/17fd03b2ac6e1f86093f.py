"""Symbolic regression: nonlinear oscillator (2) - closed-form equation."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Rational-Polynomial model with Trigonometric Drive.
    
    This approach uses a rational function for the state-dependent dynamics 
    (restoring and damping forces) to capture saturation or softening effects 
    more flexibly than a pure polynomial. The periodic driving force is 
    modeled separately with sine and cosine terms.
    
    Structure:
    y = (Numerator) / (Denominator) + Drive
    
    Numerator: p0 + p1*x + p2*v + p3*x^2 + p4*v^2 + p5*x*v
    Denominator: 1 + p6*x^2 + p7*v^2
    Drive: p8*cos(t) + p9*sin(t)
    
    Params:
    p0-p5: Coefficients for the state dynamics numerator
    p6-p7: Coefficients for the state dynamics denominator (saturation)
    p8-p9: Amplitudes for the periodic drive (cos/sin)
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # State-dependent rational term
    num = p0 + p1 * x + p2 * v + p3 * (x ** 2) + p4 * (v ** 2) + p5 * x * v
    den = 1.0 + p6 * (x ** 2) + p7 * (v ** 2)
    
    # Avoid division by zero (though 1 + positive terms is safe if p6,p7 > 0)
    # If p6 or p7 are negative, den could be small. Clip for safety.
    den = np.maximum(den, 1e-6)
    
    state_term = num / den
    
    # Periodic drive
    drive = p8 * np.cos(t) + p9 * np.sin(t)
    
    return state_term + drive
# EVOLVE-BLOCK-END
