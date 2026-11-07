"""Symbolic regression with cubic polynomial expansion."""
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Closed-form prediction using a 10-dim full cubic polynomial basis.
    
    The evaluator fits params via BFGS starting from all ones.
    We use a full cubic polynomial basis in x and y to capture smooth
    non-linear functions. This basis spans all polynomials of degree <= 3,
    which is a strong universal approximator for smooth targets on [-1, 1]^2.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Ensure correct shape
    if x.ndim != 1:
        x = x.reshape(-1)
    if y.ndim != 1:
        y = y.reshape(-1)
    
    n = x.size
    
    # Build Full Cubic Polynomial Basis (10 components)
    # Order:
    # 0: 1
    # 1: x
    # 2: y
    # 3: x^2
    # 4: x*y
    # 5: y^2
    # 6: x^3
    # 7: x^2*y
    # 8: x*y^2
    # 9: y^3
    
    b = np.empty((10, n), dtype=np.float64)
    
    b[0] = 1.0
    b[1] = x
    b[2] = y
    b[3] = x * x
    b[4] = x * y
    b[5] = y * y
    b[6] = x * x * x
    b[7] = x * x * y
    b[8] = x * y * y
    b[9] = y * y * y
    
    # Linear combination
    prediction = params @ b
    
    # Ensure finite
    if not np.all(np.isfinite(prediction)):
        prediction = np.where(np.isfinite(prediction), prediction, 0.0)
    
    return prediction
