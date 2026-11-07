"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Polynomial approximation of degree 9 (10 parameters).
    params[0:10] -> coefficients for powers 0..9 of (x/3)
    
    f(x) = sum_{i=0}^{9} c_i * (x/3)^i
    
    We scale x by 1/3 to map the domain [-3, 3] to [-1, 1].
    This improves the conditioning of the polynomial basis functions
    for the BFGS optimizer, keeping basis values bounded.
    """
    # Scale x to [-1, 1] to improve numerical stability
    t = x / 3.0
    
    # Create Vandermonde matrix for powers 0 to 9
    # Shape: (N, 10) where row i is [1, t_i, t_i^2, ..., t_i^9]
    vander = np.vander(t, N=10, increasing=True)
    
    # Linear combination of basis functions
    result = vander @ params
    
    return result
# EVOLVE-BLOCK-END
