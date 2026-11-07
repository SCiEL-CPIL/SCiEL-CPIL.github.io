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
    Predicts y using a 10-term Fourier series with fixed frequencies {1,2,3,4,5}.
    Basis: [sin(x), cos(x), sin(2x), cos(2x), sin(3x), cos(3x), sin(4x), cos(4x), sin(5x), cos(5x)]
    x: input array
    params: array of 10 coefficients
    Returns: predictions array same shape as x
    """
    if not _elapsed_ok():
        # Return a safe fallback if over deadline
        return np.zeros_like(x)
    
    # Build the Fourier basis matrix
    # Frequencies: 1, 2, 3, 4, 5
    # Order: sin(1x), cos(1x), sin(2x), cos(2x), ..., sin(5x), cos(5x)
    y = np.zeros_like(x)
    for k in range(1, 6):
        y += params[2*(k-1)] * np.sin(k * x)
        y += params[2*(k-1) + 1] * np.cos(k * x)
    
    return y
# EVOLVE-BLOCK-END
