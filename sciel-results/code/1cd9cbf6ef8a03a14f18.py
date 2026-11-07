"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Rational function model for nonlinear oscillator acceleration.
    
    Form:
    a = (p0 + p1*x + p2*v + p3*x^2 + p4*v^2 + p5*x*v + p6*sin(t) + p7*cos(t))
        / (1 + p8*x^2 + p9*v^2)
    
    This structure allows for saturation effects and complex interactions 
    that pure polynomial sums might not capture efficiently with only 10 params.
    """
    p = params
    
    # Numerator: Linear, quadratic, and cross terms in x,v plus periodic drive
    # We use sin(t) and cos(t) to capture time dependence without high-degree polynomials in t
    num = (p[0] 
           + p[1] * x 
           + p[2] * v 
           + p[3] * x**2 
           + p[4] * v**2 
           + p[5] * x * v 
           + p[6] * np.sin(t) 
           + p[7] * np.cos(t))
           
    # Denominator: 1 + quadratic terms to ensure stability and model saturation
    # x and v are small, so x**2 and v**2 are small, keeping denominator close to 1
    den = 1.0 + p[8] * x**2 + p[9] * v**2
    
    # Avoid division by zero (though unlikely with den > 0)
    # Add a small epsilon to denominator for numerical safety if it gets close to 0
    # Given start params are 1, den = 1 + x^2 + v^2 > 1.
    # BFGS might move params, but x,v are bounded.
    
    return num / den
# EVOLVE-BLOCK-END
