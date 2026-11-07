"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Basis functions to approximate a wide variety of smooth functions
    # x is in [-3, 3]
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    
    # Trigonometric terms
    sin_x = np.sin(x)
    cos_x = np.cos(x)
    sin_2x = np.sin(2 * x)
    cos_2x = np.cos(2 * x)
    
    # Gaussian bump
    exp_neg_x2 = np.exp(-x2)
    
    # Combine with parameters
    # params[0] * 1 + params[1] * x + params[2] * x^2 + ...
    return (params[0] * 1.0 + 
            params[1] * x + 
            params[2] * x2 + 
            params[3] * x3 + 
            params[4] * sin_x + 
            params[5] * cos_x + 
            params[6] * sin_2x + 
            params[7] * cos_2x + 
            params[8] * x4 + 
            params[9] * exp_neg_x2)
# EVOLVE-BLOCK-END
