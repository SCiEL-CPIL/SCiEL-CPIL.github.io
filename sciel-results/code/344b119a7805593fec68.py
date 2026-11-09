"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers the unknown function using a combination of:
    - Global polynomial trend (constant, linear, quadratic)
    - 5 localized Gaussian RBFs centered at distinct points in [-3, 3]
    - Trigonometric functions for periodic components
    
    This hybrid approach captures both global trends and local features.
    The basis is linear in parameters, so BFGS should converge quickly.
    """
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Global polynomial trend
    T0 = np.ones_like(x)
    T1 = x
    T2 = x * x
    
    # Gaussian RBFs with sigma = 0.8, centered at -2.5, -1.25, 0, 1.25, 2.5
    sigma = 0.8
    sigma2 = 2.0 * sigma * sigma
    
    G1 = np.exp(-((x + 2.5) ** 2) / sigma2)
    G2 = np.exp(-((x + 1.25) ** 2) / sigma2)
    G3 = np.exp(-(x ** 2) / sigma2)
    G4 = np.exp(-((x - 1.25) ** 2) / sigma2)
    G5 = np.exp(-((x - 2.5) ** 2) / sigma2)
    
    # Trigonometric functions
    S1 = np.sin(x)
    C1 = np.cos(x)
    
    # Linear combination
    y = (params[0] * T0 +
         params[1] * T1 +
         params[2] * T2 +
         params[3] * G1 +
         params[4] * G2 +
         params[5] * G3 +
         params[6] * G4 +
         params[7] * G5 +
         params[8] * S1 +
         params[9] * C1)
    
    return y
# EVOLVE-BLOCK-END
