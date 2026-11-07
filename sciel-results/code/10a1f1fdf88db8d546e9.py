import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a Gaussian Radial Basis Function (RBF) network.
    
    The 10 parameters are the centers of 10 Gaussian basis functions.
    The widths and amplitudes are fixed to a multi-scale set to allow
    both global (wide) and local (narrow) feature fitting.
    
    This structural change moves from a global polynomial/trig basis to a 
    localized non-linear basis, which can better handle sharp features 
    or complex shapes that global bases smooth over or fail to fit 
    due to optimization landscapes.
    """
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Ensure params has length 10
    if params.shape[0] != 10:
        # Fallback if shape is wrong, though evaluator guarantees 10
        return np.zeros_like(x, dtype=np.float64)

    # Fixed widths: 5 wide (for global trend) and 5 narrow (for local details)
    # Wide sigmas cover the domain [-2, 2] effectively
    sigma_wide = 2.0
    sigma_narrow = 0.2
    
    sigmas = np.array([
        sigma_wide, sigma_wide, sigma_wide, sigma_wide, sigma_wide,
        sigma_narrow, sigma_narrow, sigma_narrow, sigma_narrow, sigma_narrow
    ])
    
    # Fixed amplitudes: Alternating signs to allow positive and negative contributions
    # This is critical because if all amplitudes were positive, the model could 
    # only sum positive bumps, limiting its ability to fit functions with 
    # negative values or downward curvatures without a large offset.
    amplitudes = np.array([
        1.0, -1.0, 1.0, -1.0, 1.0,
        1.0, -1.0, 1.0, -1.0, 1.0
    ])
    
    # Compute RBFs
    # params[i] is the center c_i
    # term_i = a_i * exp(- (x - c_i)^2 / (2 * sigma_i^2))
    
    result = np.zeros_like(x, dtype=np.float64)
    
    for i in range(10):
        c = params[i]
        sigma = sigmas[i]
        amp = amplitudes[i]
        
        # Calculate squared distance
        diff = x - c
        # Exponent: -diff^2 / (2 * sigma^2)
        # To avoid overflow in exp for large negative values (which is fine, becomes 0)
        # or large positive (impossible for -diff^2), we are safe.
        exponent = -0.5 * (diff * diff) / (sigma * sigma)
        
        # Clip exponent to prevent underflow to 0 too early or numerical issues
        # exp(-100) is effectively 0.
        exponent = np.clip(exponent, -100.0, 0.0)
        
        result += amp * np.exp(exponent)
        
    # Ensure finite output
    if not np.all(np.isfinite(result)):
        result = np.where(np.isfinite(result), result, 0.0)
        
    return result
