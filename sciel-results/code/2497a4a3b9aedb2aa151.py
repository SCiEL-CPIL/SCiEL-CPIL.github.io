"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from numpy.polynomial import chebyshev as C

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Map input domain [-3, 3] to standard Chebyshev interval [-1, 1]
    t = x / 3.0
    
    # Compute Chebyshev basis values using numpy's optimized routine
    # V has shape (N, 10), where V[:, i] = T_i(t)
    V = C.chebvander(t, 9)
    
    # Linear combination with 10 parameters
    return V @ params
# EVOLVE-BLOCK-END
