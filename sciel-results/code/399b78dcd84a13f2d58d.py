"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression basis with 10 parameters using truncated Fourier series.
    
    Truncated Fourier basis on [-1,1]^2:
    - Cosine terms: cos(k*pi*x/2)*cos(l*pi*y/2) for k,l in {0,1,2} (9 terms)
    - One sine term: sin(pi*x/2)*sin(pi*y/2) (1 term)
    
    This captures periodic/oscillatory components that polynomial bases miss,
    while maintaining orthogonality properties for stable BFGS optimization.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    
    # Precompute trigonometric arguments
    pi_half = np.pi / 2.0
    x_arg = pi_half * x
    y_arg = pi_half * y
    
    # Cosine basis functions
    cos0_x = np.ones_like(x)
    cos1_x = np.cos(x_arg)
    cos2_x = np.cos(2.0 * x_arg)
    
    cos0_y = np.ones_like(y)
    cos1_y = np.cos(y_arg)
    cos2_y = np.cos(2.0 * y_arg)
    
    # Sine basis functions
    sin1_x = np.sin(x_arg)
    sin1_y = np.sin(y_arg)
    
    # 10 basis terms
    b1 = cos0_x * cos0_y      # (0,0) constant
    b2 = cos1_x * cos0_y      # (1,0)
    b3 = cos0_x * cos1_y      # (0,1)
    b4 = cos2_x * cos0_y      # (2,0)
    b5 = cos0_x * cos2_y      # (0,2)
    b6 = cos1_x * cos1_y      # (1,1)
    b7 = cos2_x * cos1_y      # (2,1)
    b8 = cos1_x * cos2_y      # (1,2)
    b9 = cos2_x * cos2_y      # (2,2)
    b10 = sin1_x * sin1_y     # sine cross term
    
    return (params[0] * b1 + 
            params[1] * b2 + 
            params[2] * b3 + 
            params[3] * b4 + 
            params[4] * b5 + 
            params[5] * b6 + 
            params[6] * b7 + 
            params[7] * b8 + 
            params[8] * b9 + 
            params[9] * b10)
# EVOLVE-BLOCK-END
