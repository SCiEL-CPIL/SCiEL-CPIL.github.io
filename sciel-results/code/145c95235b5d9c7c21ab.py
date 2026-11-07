"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a linear combination of diverse basis functions.
    The parameters are fitted by the evaluator using BFGS.
    """
    # Ensure x is a numpy array and handle potential shape issues
    x = np.asarray(x, dtype=np.float64)
    
    # Basis functions
    # 1. Constant
    b0 = np.ones_like(x)
    
    # 2. Linear
    b1 = x
    
    # 3. Quadratic
    b2 = x * x
    
    # 4. Cubic
    b3 = b2 * x
    
    # 5. Quartic
    b4 = b2 * b2
    
    # 6. Sine
    b5 = np.sin(x)
    
    # 7. Cosine
    b6 = np.cos(x)
    
    # 8. Tanh (Saturating odd function)
    b7 = np.tanh(x)
    
    # 9. x * tanh(x) (Odd symmetric smooth saturation, useful for asymmetric transitions)
    b8 = x * b7
    
    # 10. Gaussian bump (e^{-x^2})
    b9 = np.exp(-x * x)
    
    # Combine basis functions with parameters
    # params is an array of length 10
    p = params
    
    result = (p[0] * b0 + 
              p[1] * b1 + 
              p[2] * b2 + 
              p[3] * b3 + 
              p[4] * b4 + 
              p[5] * b5 + 
              p[6] * b6 + 
              p[7] * b7 + 
              p[8] * b8 + 
              p[9] * b9)
              
    return result
# EVOLVE-BLOCK-END
