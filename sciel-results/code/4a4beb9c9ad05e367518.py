import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximates the target function using a separable Fourier series model.
    
    Model: A*sin(a*x)*sin(b*y) + C*sin(c*x)*sin(d*y) + E*x*y + F
    
    Parameters:
    0: A (amplitude 1)
    1: a (frequency 1)
    2: b (frequency 2)
    3: C (amplitude 2)
    4: c (frequency 3)
    5: d (frequency 4)
    6: E (xy coefficient)
    7: F (constant)
    8, 9: unused (kept for compatibility, set to 0)
    
    This architecture directly targets products of sinusoids in each variable,
    which cannot be represented by linear combinations of individual sines.
    """
    # Extract parameters
    A, a, b, C, c, d, E, F = params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7]
    
    # Compute separable Fourier terms
    term1 = A * np.sin(a * x) * np.sin(b * y)
    term2 = C * np.sin(c * x) * np.sin(d * y)
    term3 = E * x * y
    term4 = F
    
    pred = term1 + term2 + term3 + term4
    
    return pred
