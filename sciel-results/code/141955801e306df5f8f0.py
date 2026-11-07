"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# Global state for caching
_cache = {}

def _build_chebyshev_basis(x, y):
    """
    Construct a basis of 10 orthogonal features using Chebyshev polynomials
    for x and y. This provides excellent numerical conditioning for BFGS.
    
    Features:
    0: 1
    1: T1(x) = x
    2: T1(y) = y
    3: T2(x) = 2x^2 - 1
    4: T2(y) = 2y^2 - 1
    5: T1(x)*T1(y) = xy
    6: T2(x)*T1(y) = (2x^2-1)y
    7: T1(x)*T2(y) = x(2y^2-1)
    8: T3(x) = 4x^3 - 3x
    9: T3(y) = 4y^3 - 3y
    """
    n = len(x)
    T1_x = x
    T1_y = y
    T2_x = 2*x**2 - 1
    T2_y = 2*y**2 - 1
    T3_x = 4*x**3 - 3*x
    T3_y = 4*y**3 - 3*y
    
    basis = np.column_stack([
        np.ones(n),
        T1_x,
        T1_y,
        T2_x,
        T2_y,
        T1_x * T1_y,
        T2_x * T1_y,
        T1_x * T2_y,
        T3_x,
        T3_y
    ])
    return basis

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Returns predictions using a fixed, well-conditioned Chebyshev basis.
    The evaluator's BFGS will fit the 10 parameters to the training data.
    """
    # Ensure inputs are arrays
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Handle edge cases
    if len(x) == 0:
        return np.array([])
    if not np.all(np.isfinite(params)):
        return np.zeros_like(y)
        
    # Build the feature matrix
    X = _build_chebyshev_basis(x, y)
    
    # If features contain non-finite values (shouldn't happen for polynomials, but safe)
    if not np.all(np.isfinite(X)):
        # Fallback to simple linear
        X = np.column_stack([np.ones(len(x)), x, y, x**2, y**2, x*y, x**3, y**3, x**2*y, x*y**2])
        if not np.all(np.isfinite(X)):
            return np.zeros_like(y)
            
    # Compute prediction
    pred = X @ params
    
    # Ensure output is finite
    if not np.all(np.isfinite(pred)):
        return np.zeros_like(y)
        
    return pred

# EVOLVE-BLOCK-END
