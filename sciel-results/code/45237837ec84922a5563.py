"""Symbolic regression with adaptive basis expansion."""
import numpy as np
import time


def _build_basis(x, y, params):
    """Build a rich basis of candidate functions from x, y and params.
    
    The first 10 parameters are linear coefficients for the basis.
    We use a diverse set of basis functions to cover common target shapes.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    
    n = x.size
    # Ensure 1D
    x = x.reshape(-1)
    y = y.reshape(-1)
    
    p = np.asarray(params, dtype=np.float64)
    # params is length 10; we use them as coefficients for a 10-dim basis
    # But we can also use params to modulate the basis itself for more flexibility
    
    # Core basis functions (10 components to match params length)
    # We design 10 basis functions that span common functional forms:
    
    b = np.empty((10, n), dtype=np.float64)
    
    # Constant
    b[0] = 1.0
    
    # Linear in x
    b[1] = x
    
    # Linear in y
    b[2] = y
    
    # Quadratic x
    b[3] = x * x
    
    # Quadratic y
    b[4] = y * y
    
    # Cross term
    b[5] = x * y
    
    # Cubic x
    b[6] = x * x * x
    
    # Cubic y
    b[7] = y * y * y
    
    # Sine of x
    b[8] = np.sin(x)
    
    # Cosine of y
    b[9] = np.cos(y)
    
    # Compute linear combination
    prediction = p @ b
    
    return prediction


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Closed-form prediction using a 10-dim basis with coefficients from params.
    
    The evaluator fits params via BFGS starting from all ones.
    We provide a rich basis so that the linear combination can approximate
    a wide variety of target functions.
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
    
    # Build basis
    b = np.empty((10, n), dtype=np.float64)
    
    b[0] = 1.0
    b[1] = x
    b[2] = y
    b[3] = x * x
    b[4] = y * y
    b[5] = x * y
    b[6] = x * x * x
    b[7] = y * y * y
    b[8] = np.sin(x)
    b[9] = np.cos(y)
    
    # Linear combination
    prediction = params @ b
    
    # Ensure finite
    if not np.all(np.isfinite(prediction)):
        prediction = np.where(np.isfinite(prediction), prediction, 0.0)
    
    return prediction
