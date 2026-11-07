"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Equation with 10 free parameters using a flexible basis:
      y = p0 + p1*x + p2*x^2 + p3*x^3 + p4*sin(x) + p5*cos(x) + p6*exp(-x) + p7*exp(-x^2) + p8*log(x+1) + p9/x
    
    This covers polynomials up to degree 3, trigonometric, decaying exponential, Gaussian,
    and reciprocal/log terms. The BFGS optimizer will fit these 10 parameters.
    """
    x = np.asarray(x, dtype=float)
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Polynomial terms
    term1 = p0
    term2 = p1 * x
    term3 = p2 * x**2
    term4 = p3 * x**3
    
    # Trigonometric terms
    term5 = p4 * np.sin(x)
    term6 = p5 * np.cos(x)
    
    # Exponential terms (decaying to avoid overflow)
    term7 = p6 * np.exp(-x)
    term8 = p7 * np.exp(-x**2)
    
    # Log and reciprocal terms
    term9 = p8 * np.log(x + 1.0)
    term10 = p9 / (x + 0.1)
    
    return term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9 + term10
# EVOLVE-BLOCK-END
