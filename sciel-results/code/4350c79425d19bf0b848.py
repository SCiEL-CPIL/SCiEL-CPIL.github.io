import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Complete polynomial basis of degree 3.
    This provides a flexible, linear-in-parameters model that can capture
    quadratic and cubic nonlinearities in position and velocity, as well as
    their interactions. This avoids the ill-conditioning of highly correlated
    non-linear basis functions (like sin/tanh near zero) and allows the
    optimizer to find a robust global minimum.
    """
    # Pre-compute powers and interactions
    x2 = x * x
    x3 = x2 * x
    v2 = v * v
    v3 = v2 * v
    
    # Basis terms: 1, x, v, x^2, v^2, xv, x^3, v^3, x^2v, xv^2
    # Mapping to params[0] through params[9]
    
    return (params[0] + 
            params[1] * x + 
            params[2] * v + 
            params[3] * x2 + 
            params[4] * v2 + 
            params[5] * x * v + 
            params[6] * x3 + 
            params[7] * v3 + 
            params[8] * x2 * v + 
            params[9] * x * v2)
