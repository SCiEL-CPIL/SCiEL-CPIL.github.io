"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Flatten inputs to ensure consistent shape
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    
    # Compute basis features
    # We use a complete polynomial basis of degree 3 in x and y.
    # This allows the model to capture non-linear interactions and curvatures
    # that a linear model (degree 1) would miss.
    
    # Feature 1: Bias
    f1 = np.ones_like(x)
    
    # Feature 2: x
    f2 = x
    
    # Feature 3: y
    f3 = y
    
    # Feature 4: x^2
    f4 = x * x
    
    # Feature 5: y^2
    f5 = y * y
    
    # Feature 6: x * y
    f6 = x * y
    
    # Feature 7: x^3
    f7 = x * x * x
    
    # Feature 8: y^3
    f8 = y * y * y
    
    # Feature 9: x^2 * y
    f9 = x * x * y
    
    # Feature 10: x * y^2
    f10 = x * y * y
    
    # Linear combination of features with the given parameters
    # params is expected to be a 1D array of length 10
    pred = (params[0] * f1 + 
            params[1] * f2 + 
            params[2] * f3 + 
            params[3] * f4 + 
            params[4] * f5 + 
            params[5] * f6 + 
            params[6] * f7 + 
            params[7] * f8 + 
            params[8] * f9 + 
            params[9] * f10)
            
    return pred
# EVOLVE-BLOCK-END
