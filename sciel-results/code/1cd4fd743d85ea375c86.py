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
    Predicts y using a Fourier series with 5 harmonics.
    x: input array
    params: array of 10 coefficients [A1, B1, A2, B2, A3, B3, A4, B4, A5, B5]
    Returns: predictions array same shape as x
    """
    if not _elapsed_ok():
        # Return a safe fallback if over deadline
        return np.zeros_like(x)
    
    # x is in [-3, 3]. Frequencies 1 to 5.
    # We can scale x to make frequencies more effective if needed, but let's stick to standard.
    # y = sum_{k=1}^5 (A_k * cos(k*x) + B_k * sin(k*x))
    
    y = np.zeros_like(x)
    
    # Unpack params
    # params: [A1, B1, A2, B2, A3, B3, A4, B4, A5, B5]
    
    for k in range(5):
        Ak = params[2*k]
        Bk = params[2*k + 1]
        angle = (k + 1) * x
        y += Ak * np.cos(angle) + Bk * np.sin(angle)
        
    return y
# EVOLVE-BLOCK-END
