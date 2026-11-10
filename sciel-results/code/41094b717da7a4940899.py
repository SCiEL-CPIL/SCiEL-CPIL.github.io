"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # New approach: Radial Basis Functions (Gaussian) with fixed centers
    # This allows for local features and non-smooth approximations, 
    # unlike the global polynomial.
    
    # 5 centers covering the domain [-2, 2]
    centers = np.array([-1.8, -0.9, 0.0, 0.9, 1.8])
    
    # params[0:5] are amplitudes A_i
    # params[5:10] are log-widths. Widths w_i = exp(params[5:10])
    amplitudes = params[0:5]
    log_widths = params[5:10]
    widths = np.exp(log_widths)
    
    # Ensure widths are not too small to avoid numerical issues, 
    # though exp() keeps them positive. 
    # We might want to clamp widths to a minimum to prevent division by zero 
    # in the exponent if width is extremely small, but exp() handles it.
    # However, if width is very small, the Gaussian becomes a spike.
    # Let's enforce a minimum width of 0.01 for stability.
    widths = np.maximum(widths, 0.01)
    
    # Vectorized computation
    # x is shape (N,)
    # centers is shape (5,)
    # We want to compute sum over i of A_i * exp(-(x - c_i)^2 / w_i^2)
    
    # Reshape for broadcasting:
    # x -> (N, 1)
    # centers -> (1, 5)
    # widths -> (1, 5)
    # amplitudes -> (1, 5)
    
    x_col = x[:, np.newaxis]
    c_row = centers[np.newaxis, :]
    w_row = widths[np.newaxis, :]
    a_row = amplitudes[np.newaxis, :]
    
    # Calculate squared distances
    diff = x_col - c_row
    sq_dist = diff ** 2
    
    # Calculate Gaussian terms
    # Note: w^2 in denominator
    gaussian_terms = np.exp(-sq_dist / (w_row ** 2))
    
    # Weight by amplitudes and sum
    result = np.sum(gaussian_terms * a_row, axis=1)
    
    return result
# EVOLVE-BLOCK-END
