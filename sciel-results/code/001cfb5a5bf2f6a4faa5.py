"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = None
_TIME_LIMIT = 115.0  # Safety margin before 120s kill

def _check_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + _TIME_LIMIT
    return time.time() < _DEADLINE


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximate the target function using a mixed basis of polynomial,
    trigonometric, exponential, and rational terms.
    
    Args:
        x: Input array of shape (N,).
        params: Array of 10 parameters.
        
    Returns:
        Prediction array of shape (N,).
    """
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Check deadline
    if not _check_deadline():
        # Return a simple fallback if we're running out of time
        return np.ones_like(x) * params[0]
    
    n = len(x)
    basis = np.empty((n, 10), dtype=np.float64)
    
    # Polynomial terms: 1, x, x^2
    basis[:, 0] = 1.0
    basis[:, 1] = x
    basis[:, 2] = x ** 2
    
    # Trig terms: sin(x), cos(x)
    basis[:, 3] = np.sin(x)
    basis[:, 4] = np.cos(x)
    
    # Exponential terms: exp(-x), exp(x)
    basis[:, 5] = np.exp(-x)
    basis[:, 6] = np.exp(x)
    
    # Rational terms: 1/x, 1/(1+x)
    basis[:, 7] = 1.0 / x
    basis[:, 8] = 1.0 / (1.0 + x)
    
    # Log term
    basis[:, 9] = np.log(x)
    
    # Check for non-finite values and handle gracefully
    if not np.all(np.isfinite(basis)):
        # Replace non-finite values with 0
        basis = np.nan_to_num(basis, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Prediction is the linear combination of the basis functions
    result = basis @ params
    
    # Ensure output is finite
    if not np.all(np.isfinite(result)):
        result = np.nan_to_num(result, nan=0.0, posinf=1e10, neginf=-1e10)
    
    return result
# EVOLVE-BLOCK-END
