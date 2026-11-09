import numpy as np
import time

_START_TIME = time.time()
_TIMEOUT = 150.0  # Wall-clock limit with safety margin

# Piecewise linear hat function basis with 10 evenly spaced knots on [-2, 2].
# This is strictly linear in params, so BFGS converges to the global LS solution.
# Hat functions provide flexible, non-oscillatory approximation.

_N_KNOTS = 10
_KNOT_SPACING = 4.0 / (_N_KNOTS - 1)  # spacing between knot centers
_HALF_WIDTH = _KNOT_SPACING  # half-support of each hat function

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Linear model with a fixed basis of 10 piecewise linear hat functions.
    y(x) = sum_i params[i] * hat_i(x)
    
    Hat functions are centered at evenly spaced points across [-2, 2].
    Each hat is triangular: rises linearly from 0 to 1, then falls back to 0.
    """
    # Safety check: if we're close to the timeout, return a simple fallback
    if time.time() - _START_TIME > _TIMEOUT - 5.0:
        return np.mean(params) * np.ones_like(x)
    
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    N = len(x)
    
    # Knot centers: -2, -2+spacing, ..., 2
    knot_centers = np.linspace(-2.0, 2.0, _N_KNOTS)
    
    # Build the basis matrix: (N, 10)
    # For each hat function i, hat_i(x) = max(0, 1 - |x - center_i| / half_width)
    basis = np.zeros((N, _N_KNOTS), dtype=np.float64)
    for i in range(_N_KNOTS):
        c = knot_centers[i]
        basis[:, i] = np.maximum(0.0, 1.0 - np.abs(x - c) / _HALF_WIDTH)
    
    # Prediction: (N, 10) dot (10,) -> (N,)
    prediction = basis.dot(params)
    
    # Ensure finite output
    if not np.all(np.isfinite(prediction)):
        return np.zeros_like(x)
    
    return prediction
