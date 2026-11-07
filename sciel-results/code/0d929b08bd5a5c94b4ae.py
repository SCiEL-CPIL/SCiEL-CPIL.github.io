import numpy as np
import time

# Module-level cache for basis matrix (computed once per unique x,y grid)
_basis_cache = {}

def _compute_basis(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute a 200x10 basis matrix using a rich set of low-order polynomial and
    trigonometric features. The basis is designed to be well-conditioned and
    span a large function space for symbolic regression on [-1,1]^2.
    
    Features:
    0: 1 (bias)
    1: x
    2: y
    3: x^2
    4: y^2
    5: x*y
    6: x^3
    7: y^3
    8: x^2*y
    9: x*y^2
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    
    # Use a hashable key for caching
    key = (x.tobytes(), y.tobytes())
    if key in _basis_cache:
        return _basis_cache[key]
    
    # Build basis columns
    b0 = np.ones_like(x)
    b1 = x
    b2 = y
    b3 = x * x
    b4 = y * y
    b5 = x * y
    b6 = x * x * x
    b7 = y * y * y
    b8 = x * x * y
    b9 = x * y * y
    
    B = np.column_stack([b0, b1, b2, b3, b4, b5, b6, b7, b8, b9])
    
    # Cache it
    _basis_cache[key] = B
    return B


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Equation for symbolic regression with 2 inputs x, y.
    Uses a polynomial basis of degree up to 3 with 10 coefficients.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    p = np.asarray(params, dtype=np.float64)
    
    if p.shape != (10,):
        p = p.reshape(10)
    
    B = _compute_basis(x, y)
    pred = B @ p
    
    # Safety check
    if not np.all(np.isfinite(pred)):
        pred = np.where(np.isfinite(pred), pred, 0.0)
    
    return pred
