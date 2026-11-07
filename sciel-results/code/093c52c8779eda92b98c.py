"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression using a Fourier series basis.
    This allows the model to fit periodic or oscillatory functions.
    
    params: array of length 10, representing coefficients for:
    [1, cos(t), cos(2t), cos(3t), cos(4t), sin(t), sin(2t), sin(3t), sin(4t), sin(5t)]
    where t = pi * x / 3
    """
    x = np.asarray(x, dtype=float)
    
    # Map x from [-3, 3] to [-pi, pi]
    t = np.pi * x / 3.0
    
    # Precompute trigonometric values
    # We need cos(1t)..cos(4t) and sin(1t)..sin(5t)
    # Note: cos(0t) = 1 is the first parameter
    
    c1 = np.cos(t)
    c2 = np.cos(2 * t)
    c3 = np.cos(3 * t)
    c4 = np.cos(4 * t)
    
    s1 = np.sin(t)
    s2 = np.sin(2 * t)
    s3 = np.sin(3 * t)
    s4 = np.sin(4 * t)
    s5 = np.sin(5 * t)
    
    # Construct the prediction as a linear combination
    # params[0] * 1 + params[1] * c1 + ... + params[9] * s5
    
    pred = (params[0] * 1.0 + 
            params[1] * c1 + 
            params[2] * c2 + 
            params[3] * c3 + 
            params[4] * c4 + 
            params[5] * s1 + 
            params[6] * s2 + 
            params[7] * s3 + 
            params[8] * s4 + 
            params[9] * s5)
            
    return pred
# EVOLVE-BLOCK-END
