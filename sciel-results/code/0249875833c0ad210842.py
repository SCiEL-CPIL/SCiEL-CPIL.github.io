"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Uses a basis of Fourier (trigonometric) functions.
    This approach is distinct from polynomial bases and is effective for
    capturing smooth, potentially periodic or oscillatory structures in
    the bounded domain [-1, 1].
    """
    # Basis functions
    # 1. Constant
    b0 = np.ones_like(x)
    
    # 2-3. First harmonic in x
    b1 = np.sin(x)
    b2 = np.cos(x)
    
    # 4-5. First harmonic in y
    b3 = np.sin(y)
    b4 = np.cos(y)
    
    # 6-9. Interactions of first harmonics
    b5 = np.sin(x) * np.sin(y)
    b6 = np.sin(x) * np.cos(y)
    b7 = np.cos(x) * np.sin(y)
    b8 = np.cos(x) * np.cos(y)
    
    # 10. Second harmonic in x (or interaction)
    # Using sin(2*x) to capture higher frequency curvature in x
    b9 = np.sin(2 * x)
    
    # Combine with parameters
    pred = (params[0] * b0 + 
            params[1] * b1 + 
            params[2] * b2 + 
            params[3] * b3 + 
            params[4] * b4 + 
            params[5] * b5 + 
            params[6] * b6 + 
            params[7] * b7 + 
            params[8] * b8 + 
            params[9] * b9)
            
    return pred
# EVOLVE-BLOCK-END
