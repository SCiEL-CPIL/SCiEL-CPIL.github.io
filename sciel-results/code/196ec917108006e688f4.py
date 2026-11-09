import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Combined Chebyshev + Gaussian RBF basis for symbolic regression.
    Uses 5 Chebyshev polynomials (T0..T4) and 5 Gaussian RBFs with fixed centers.
    This linear-in-params model is well-conditioned for BFGS fitting.
    """
    # Normalize input to [-1, 1] for Chebyshev polynomials
    t = x / 3.0
    
    # Chebyshev polynomials of the first kind T_n(t)
    T0 = np.ones_like(t)
    T1 = t
    T2 = 2 * t * T1 - T0
    T3 = 2 * t * T2 - T1
    T4 = 2 * t * T3 - T2
    
    # Gaussian RBF terms with fixed centers in [-1, 1]
    # Centers: -0.8, -0.4, 0.0, 0.4, 0.8
    sigma = 0.6
    c1 = -0.8
    c2 = -0.4
    c3 = 0.0
    c4 = 0.4
    c5 = 0.8
    
    G1 = np.exp(-((t - c1) / sigma) ** 2)
    G2 = np.exp(-((t - c2) / sigma) ** 2)
    G3 = np.exp(-((t - c3) / sigma) ** 2)
    G4 = np.exp(-((t - c4) / sigma) ** 2)
    G5 = np.exp(-((t - c5) / sigma) ** 2)
    
    # Combine with parameters
    # params[0..4] for Chebyshev, params[5..9] for Gaussians
    y = (params[0] * T0 + 
         params[1] * T1 + 
         params[2] * T2 + 
         params[3] * T3 + 
         params[4] * T4 +
         params[5] * G1 + 
         params[6] * G2 + 
         params[7] * G3 + 
         params[8] * G4 + 
         params[9] * G5)
         
    return y
