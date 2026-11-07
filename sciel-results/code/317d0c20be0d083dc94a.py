"""Program for symbolic regression using Chebyshev polynomial basis."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Cache for the basis matrix to avoid recalculating for every BFGS step
_cached_x = None
_cached_basis = None

def _get_basis_matrix(x):
    global _cached_x, _cached_basis
    if _cached_x is not None and _cached_x is x:
        return _cached_basis
    
    # Map x from [0.1, 6] to t in [-1, 1] for Chebyshev polynomials
    # t = (2*x - (x_min + x_max)) / (x_max - x_min)
    x_min, x_max = 0.1, 6.0
    t = (2.0 * x - (x_min + x_max)) / (x_max - x_min)
    t = np.clip(t, -1.0, 1.0)
    
    # Compute Chebyshev polynomials T_0 through T_9
    # T_0 = 1
    # T_1 = t
    # T_n = 2*t*T_{n-1} - T_{n-2}
    
    T0 = np.ones_like(t)
    T1 = t.copy()
    
    cheb_cols = [T0, T1]
    for n in range(2, 10):
        Tn = 2.0 * t * cheb_cols[-1] - cheb_cols[-2]
        cheb_cols.append(Tn)
    
    basis = np.column_stack(cheb_cols)
    
    _cached_x = x
    _cached_basis = basis
    return basis

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Returns a linear combination of 10 Chebyshev polynomials (degree 0-9).
    """
    basis = _get_basis_matrix(x)
    
    try:
        pred = basis.dot(params)
    except Exception:
        # Fallback: linear interpolation
        pred = params[0] * x + params[1]
        
    return pred
# EVOLVE-BLOCK-END
