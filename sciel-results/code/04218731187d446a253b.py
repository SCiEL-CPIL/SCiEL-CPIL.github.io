"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # 9th-degree polynomial basis on scaled input u = x/3
    # This is a linear model in the parameters, allowing BFGS to find the global least-squares solution.
    # params[0..9] are the coefficients for u^0 to u^9
    
    u = x / 3.0
    
    # Use Horner's method for numerical stability and efficiency
    # y = c0 + u(c1 + u(c2 + ... + u(c9)))
    result = np.zeros_like(x)
    for i in range(9, -1, -1):
        result = result * u + params[i]
        
    return result
# EVOLVE-BLOCK-END
