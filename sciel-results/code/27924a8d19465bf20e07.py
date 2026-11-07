"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Radial Basis Function (RBF) model with 5 centers and 5 amplitudes.
    Fixed Gaussian width sigma=1.0.
    params[0:5] are centers c_i
    params[5:10] are amplitudes A_i
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    # Unpack parameters
    # Centers
    c1, c2, c3, c4, c5 = params[0], params[1], params[2], params[3], params[4]
    # Amplitudes
    A1, A2, A3, A4, A5 = params[5], params[6], params[7], params[8], params[9]

    # Fixed width
    sigma = 1.0
    
    # Compute Gaussian kernels
    # Using exp(-((x-c)^2) / (2*sigma^2))
    # To avoid overflow/underflow issues, we clip the exponent argument
    # although for sigma=1 and x in [-2,2], it's safe.
    
    k1 = np.exp(-((x - c1) ** 2) / (2 * sigma ** 2))
    k2 = np.exp(-((x - c2) ** 2) / (2 * sigma ** 2))
    k3 = np.exp(-((x - c3) ** 2) / (2 * sigma ** 2))
    k4 = np.exp(-((x - c4) ** 2) / (2 * sigma ** 2))
    k5 = np.exp(-((x - c5) ** 2) / (2 * sigma ** 2))
    
    # Linear combination
    y = A1 * k1 + A2 * k2 + A3 * k3 + A4 * k4 + A5 * k5
    
    # Ensure finite output
    if not np.all(np.isfinite(y)):
        # If non-finite, return zeros to allow optimizer to proceed (though it will penalize)
        # Actually, returning zeros is better than crashing.
        y = np.zeros_like(x)
        
    return y
# EVOLVE-BLOCK-END
