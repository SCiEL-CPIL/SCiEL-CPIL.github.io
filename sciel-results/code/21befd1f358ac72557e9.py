"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Non-linear symbolic regression model using 10 Gaussian Radial Basis Functions.
    
    The 10 parameters are grouped into 5 pairs: (amplitude, center) for 5 RBFs.
    The width of each Gaussian is fixed at 0.5 to balance flexibility and conditioning.
    This allows the model to fit non-linear, non-polynomial functions (e.g., exp, sin, step-like)
    which a linear basis of degree 9 would struggle to approximate accurately without overfitting or ill-conditioning.
    """
    # Unpack parameters into 5 pairs of (amplitude, center)
    # params[0], params[1] -> amp1, center1
    # params[2], params[3] -> amp2, center2
    # ...
    # params[8], params[9] -> amp5, center5
    
    # Fixed width to ensure stability and prevent collapse to a single spike
    width = 0.5
    inv_width_sq = 1.0 / (2.0 * width * width)
    
    y = np.zeros_like(x, dtype=np.float64)
    
    for i in range(5):
        amp = params[2 * i]
        center = params[2 * i + 1]
        
        # Gaussian RBF: amp * exp(-((x - center)^2) / (2 * width^2))
        # Using np.exp directly for clarity and standard math
        diff = x - center
        y += amp * np.exp(-diff * diff * inv_width_sq)
        
    return y
# EVOLVE-BLOCK-END
