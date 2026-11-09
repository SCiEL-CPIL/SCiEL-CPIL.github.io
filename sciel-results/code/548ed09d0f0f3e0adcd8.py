"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression candidate using a Radial Basis Function (RBF) network.
    
    Model:
    y = p0 + 
        p1 * exp(-((x - p2)^2) / (2 * sigma1^2)) +
        p3 * exp(-((x - p4)^2) / (2 * sigma2^2)) +
        p5 * exp(-((x - p6)^2) / (2 * sigma3^2))
        
    Where sigma_i = exp(p_width_i). This ensures positive widths and stable gradients.
    We use 3 Gaussian components and a bias term. Total 10 parameters.
    
    Parameters:
    p0: Bias (offset)
    p1: Amplitude of Gaussian 1
    p2: Center of Gaussian 1
    p3: Log-Width of Gaussian 1 (sigma1 = exp(p3))
    p4: Amplitude of Gaussian 2
    p5: Center of Gaussian 2
    p6: Log-Width of Gaussian 2 (sigma2 = exp(p6))
    p7: Amplitude of Gaussian 3
    p8: Center of Gaussian 3
    p9: Log-Width of Gaussian 3 (sigma3 = exp(p9))
    """
    # Extract parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Calculate widths using exp to ensure positivity and non-zero values
    # Initial params are 1, so sigma = exp(1) ~ 2.718, which is a good scale for [-3, 3]
    sigma1 = np.exp(p3)
    sigma2 = np.exp(p6)
    sigma3 = np.exp(p9)
    
    # Calculate Gaussian terms
    # Clipping the exponent to prevent underflow/overflow issues, though exp handles it mostly
    # (x-c)^2 / (2*sigma^2) can be large negative, exp(large neg) -> 0. This is fine.
    # We just need to ensure sigma is not 0. exp() is never 0.
    
    term1 = p1 * np.exp(-((x - p2) ** 2) / (2.0 * sigma1 ** 2))
    term2 = p4 * np.exp(-((x - p5) ** 2) / (2.0 * sigma2 ** 2))
    term3 = p7 * np.exp(-((x - p8) ** 2) / (2.0 * sigma3 ** 2))
    
    return p0 + term1 + term2 + term3
# EVOLVE-BLOCK-END
