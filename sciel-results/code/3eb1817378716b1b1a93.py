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
    Predicts y using a 10-term Fourier series with frequencies 1-5.
    The basis is strictly linear in params, creating a well-conditioned
    system for BFGS to solve exactly.
    
    x: input array
    params: array of 10 coefficients
    Returns: predictions array same shape as x
    """
    if not _elapsed_ok():
        # Return a safe fallback if over deadline
        return np.zeros_like(x)
    
    x = np.asarray(x, dtype=float)
    
    # Compute the 10 Fourier basis functions
    # Frequencies: 1, 2, 3, 4, 5
    # For each frequency k, we have sin(k*pi*x/3) and cos(k*pi*x/3)
    # This gives 10 basis functions total
    
    # Precompute the scaled argument for efficiency
    # We use x directly, scaled so that the period covers [-3, 3]
    # sin(k * pi * x / 3) has period 6/k, so over [-3,3] we see k/2 periods
    
    basis = np.empty((x.shape[0], 10), dtype=float)
    
    for k in range(1, 6):
        arg = k * np.pi * x / 3.0
        basis[:, 2*(k-1)] = np.sin(arg)
        basis[:, 2*(k-1)+1] = np.cos(arg)
    
    # Compute prediction as linear combination
    y = basis @ params
    
    # Ensure finite output
    if not np.all(np.isfinite(y)):
        return np.zeros_like(x)
        
    return y
# EVOLVE-BLOCK-END
