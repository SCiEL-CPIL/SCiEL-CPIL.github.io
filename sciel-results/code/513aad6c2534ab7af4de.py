"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Fourier series approximation with 5 harmonics (10 parameters).
    params[0:5]  -> cosine coefficients for harmonics 1..5
    params[5:10] -> sine coefficients for harmonics 1..5
    
    f(x) = sum_{k=1}^{5} [ a_k * cos(k * pi * x / 3) + b_k * sin(k * pi * x / 3) ]
    
    We scale x by pi/3 so that the fundamental period covers the range [-3, 3],
    making the basis well-conditioned for BFGS optimization.
    """
    # Scale x to make the fundamental period match the domain [-3, 3]
    # This ensures the basis functions are well-behaved over the input range
    x_scaled = np.pi * x / 3.0
    
    # Extract coefficients
    a = params[0:5]  # cosine coefficients
    b = params[5:10]  # sine coefficients
    
    # Compute harmonics k = 1, 2, 3, 4, 5
    result = np.zeros_like(x, dtype=np.float64)
    
    for k in range(1, 6):
        result += a[k-1] * np.cos(k * x_scaled)
        result += b[k-1] * np.sin(k * x_scaled)
    
    return result
# EVOLVE-BLOCK-END
