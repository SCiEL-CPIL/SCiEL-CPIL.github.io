"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard: track start time at module level
_START_TIME = time.time()
_DEADLINE = 140.0  # Safety margin before 150s kill


def _elapsed_ok():
    """Check if we're still within deadline."""
    return (time.time() - _START_TIME) < _DEADLINE


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Predicts y using a degree-9 Chebyshev polynomial fit.
    x: input array
    params: array of 10 coefficients
    Returns: predictions array same shape as x
    """
    if not _elapsed_ok():
        # Return a safe fallback if over deadline
        return np.zeros_like(x)
    
    # Scale x from [-3, 3] to [-1, 1] for standard Chebyshev polynomials
    t = x / 3.0
    
    # Use numpy's built-in Chebyshev evaluation for better numerical stability
    y = np.polynomial.chebyshev.chebval(t, params)
    
    return y
# EVOLVE-BLOCK-END
