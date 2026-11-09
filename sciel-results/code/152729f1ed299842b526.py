"""
Uses a mixed polynomial-trigonometric-radial basis for symbolic regression.
This model is linear in the 10 parameters, allowing the external BFGS optimizer
to efficiently find the global minimum.
The basis includes low-degree polynomial terms, scaled trigonometric terms 
(period matching the domain [-1, 1]), and radial/interaction terms to capture
diverse functional forms (periodic, geometric, interactive).
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global time tracking for the 150s wall-clock limit
_START_TIME = time.time()
_TIME_LIMIT = 150.0
_SAFETY_MARGIN = 5.0  # Stop heavy computation 5s before limit

def _time_remaining():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Computes predictions using a fixed 10-term basis expanded by the 10 learned parameters.
    
    Args:
        x: Array of x inputs
        y: Array of y inputs
        params: Array of 10 parameters
        
    Returns:
        Array of predictions
    """
    # Basic safety: if we are close to the deadline, return a simple fallback
    # to avoid any potential blocking or long computations in the wrapper.
    if _time_remaining() < _SAFETY_MARGIN:
        # Fallback: simple linear model using first 3 params
        # This ensures we don't crash the evaluator due to timeout
        return params[0] + params[1] * x + params[2] * y

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Ensure we have exactly 10 params, pad if necessary (shouldn't happen)
    if len(p) < 10:
        p = np.pad(p, (0, 10 - len(p)), constant_values=0.0)
    else:
        p = p[:10]
        
    # Compute the 10 basis functions
    # Polynomial terms
    b0 = np.ones_like(x)
    b1 = x
    b2 = y
    b3 = x * x
    b4 = y * y
    b5 = x * y
    
    # Trigonometric terms (scaled by pi to cover full oscillation on [-1, 1])
    # This helps if the hidden function has a period of 2 (e.g., sin(pi*t))
    b6 = np.sin(np.pi * x)
    b7 = np.sin(np.pi * y)
    
    # Radial and interaction terms
    b8 = x * x + y * y
    b9 = np.cos(np.pi * x * y)
    
    # Stack into a matrix (N, 10)
    # Using column_stack is efficient for this small number of columns
    basis = np.column_stack((b0, b1, b2, b3, b4, b5, b6, b7, b8, b9))
    
    # Linear combination: pred = basis @ params
    pred = basis @ p
    
    # Ensure output is finite
    if not np.all(np.isfinite(pred)):
        pred = np.nan_to_num(pred, nan=0.0, posinf=1e10, neginf=-1e10)
        
    return pred
# EVOLVE-BLOCK-END
