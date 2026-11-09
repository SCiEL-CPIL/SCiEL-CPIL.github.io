"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid model: combines a global polynomial basis (degree 4) with local
    piecewise-linear hat functions. The model is strictly linear in params,
    so BFGS converges to the global least-squares minimum.
    
    Basis functions:
    1. Polynomial terms: 1, x, x^2, x^3, x^4 (5 parameters)
    2. Hat functions: 5 evenly spaced hats across [-2, 2] (5 parameters)
    
    Total: 10 parameters.
    """
    # Polynomial basis (indices 0-4)
    p0 = params[0]
    p1 = params[1]
    p2 = params[2]
    p3 = params[3]
    p4 = params[4]
    
    result = p0 + p1 * x + p2 * x**2 + p3 * x**3 + p4 * x**4
    
    # Hat function basis (indices 5-9)
    # Define 5 hat centers evenly spaced in [-2, 2]
    # Centers: -2, -1, 0, 1, 2
    # Width: 2 (so each hat spans 2 units)
    # Hat i: max(0, 1 - |x - c_i| / w) where w is half-width
    # For centers -2, -1, 0, 1, 2 with half-width 1:
    #   hat_0: max(0, 1 - |x - (-2)|) -> nonzero on [-3, -1]
    #   hat_1: max(0, 1 - |x - (-1)|) -> nonzero on [-2, 0]
    #   hat_2: max(0, 1 - |x - 0|)    -> nonzero on [-1, 1]
    #   hat_3: max(0, 1 - |x - 1|)    -> nonzero on [0, 2]
    #   hat_4: max(0, 1 - |x - 2|)    -> nonzero on [1, 3]
    
    centers = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    half_width = 1.0
    
    for i, c in enumerate(centers):
        hat_val = np.maximum(0.0, 1.0 - np.abs(x - c) / half_width)
        result += params[5 + i] * hat_val
    
    return result
# EVOLVE-BLOCK-END
