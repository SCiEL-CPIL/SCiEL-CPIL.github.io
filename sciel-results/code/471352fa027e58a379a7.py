"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis: polynomial (degree 5) + two sine terms with learnable frequency.
    
    Parameters (length 10):
    p0: constant
    p1: x (linear)
    p2: x^2
    p3: x^3
    p4: x^4
    p5: x^5
    p6: amplitude of sin(f1*x)
    p7: frequency f1 of first sine
    p8: amplitude of sin(f2*x)
    p9: frequency f2 of second sine
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    
    # Scale to [-1, 1] for numerical stability of high-order polynomials
    z = x / 2.0
    
    # Polynomial part (degree 5) in scaled variable z
    z2 = z * z
    z3 = z2 * z
    z4 = z2 * z2
    z5 = z4 * z
    
    y = params[0] + params[1] * z + params[2] * z2 + params[3] * z3 + params[4] * z4 + params[5] * z5
    
    # Sine terms with learnable frequencies
    # Clip frequencies to avoid extreme oscillation and numerical issues
    f1 = np.clip(params[6], -50.0, 50.0)
    f2 = np.clip(params[7], -50.0, 50.0)
    
    # Amplitudes
    amp1 = params[8]
    amp2 = params[9]
    
    y += amp1 * np.sin(f1 * x)
    y += amp2 * np.sin(f2 * x)
    
    return y
# EVOLVE-BLOCK-END
