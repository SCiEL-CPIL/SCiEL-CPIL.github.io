"""Program for symbolic regression using a linear combination of basis functions."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Cache for the basis matrix to avoid recalculating for every BFGS step
_cached_x = None
_cached_basis = None

def _get_basis_matrix(x):
    global _cached_x, _cached_basis
    # Simple check if x has changed (identity check or shape/value)
    # Since x is fixed during fitting, this is safe.
    if _cached_x is not None and _cached_x is x:
        return _cached_basis
    
    # Handle first call or change in x
    # Basis functions with scaling to improve condition number
    # x range is [0.1, 6]
    
    x_safe = np.clip(x, 1e-6, None)
    
    # Scale factors to bring magnitudes closer to O(1)
    # 1: 1
    # x: mean ~3 -> /3
    # x^2: mean ~10 -> /10
    # x^3: mean ~35 -> /30
    # x^4: mean ~115 -> /100
    # 1/x: mean ~0.6 -> *5
    # log(x): mean ~1.1 -> *1
    # sin(x): std ~0.7 -> *1
    # cos(x): std ~0.7 -> *1
    # exp(x): mean ~100 -> /100
    
    basis = np.column_stack((
        np.ones_like(x_safe),
        x_safe / 3.0,
        (x_safe**2) / 10.0,
        (x_safe**3) / 30.0,
        (x_safe**4) / 100.0,
        (1.0 / x_safe) * 5.0,
        np.log(x_safe),
        np.sin(x_safe),
        np.cos(x_safe),
        np.exp(x_safe) / 100.0
    ))
    
    _cached_x = x
    _cached_basis = basis
    return basis

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Returns a linear combination of 10 scaled basis functions.
    """
    basis = _get_basis_matrix(x)
    
    try:
        pred = basis.dot(params)
    except Exception:
        # Fallback
        pred = params[0] * x + params[1]
        
    return pred
# EVOLVE-BLOCK-END
