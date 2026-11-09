# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression with rich basis for better BFGS convergence.
    
    Uses 10 well-chosen basis functions that span a rich function space
    while maintaining good conditioning for the linear fit.
    
    Basis functions are scaled to have roughly unit norm over [-3, 3]
    to improve the condition number of the design matrix, helping BFGS
    converge more reliably.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Define 10 basis functions with improved conditioning
    # Scale polynomial terms to reduce magnitude differences
    
    # Basis 0: constant
    b0 = np.ones_like(x)
    
    # Basis 1: linear (scaled)
    b1 = x / 3.0
    
    # Basis 2: quadratic (scaled)
    b2 = (x / 3.0)**2
    
    # Basis 3: cubic (scaled)
    b3 = (x / 3.0)**3
    
    # Basis 4: quartic (scaled) - captures even symmetric behavior
    b4 = (x / 3.0)**4
    
    # Basis 5: sin(x)
    b5 = np.sin(x)
    
    # Basis 6: cos(x)
    b6 = np.cos(x)
    
    # Basis 7: sin(2x)
    b7 = np.sin(2 * x)
    
    # Basis 8: cos(2x)
    b8 = np.cos(2 * x)
    
    # Basis 9: Gaussian bump centered at 0
    b9 = np.exp(-0.5 * x**2)
    
    # Stack basis functions: shape (n, 10)
    basis = np.column_stack([b0, b1, b2, b3, b4, b5, b6, b7, b8, b9])
    
    # Linear combination with the 10 parameters
    pred = basis @ params
    
    return pred
# EVOLVE-BLOCK-END
