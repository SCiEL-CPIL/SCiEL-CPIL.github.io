"""Symbolic regression with Legendre polynomial basis."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Evaluate a degree-9 Legendre polynomial in x.
    
    x: input array (values in [0.1, 6])
    params: array of length 10, coefficients for P_0 through P_9
    
    Returns: array of predictions, one per point in x.
    """
    # Map x from [0.1, 6] to [-1, 1]
    x_min = 0.1
    x_max = 6.0
    t = (2.0 * (x - x_min) / (x_max - x_min)) - 1.0
    
    # Build Legendre basis matrix P of shape (n, 10)
    # P[:, k] = P_k(t) for k = 0, ..., 9
    n = len(x)
    P = np.zeros((n, 10), dtype=np.float64)
    P[:, 0] = 1.0
    if n > 0:
        P[:, 1] = t
    for k in range(2, 10):
        # Recurrence: P_k(t) = ((2k-1)t P_{k-1}(t) - (k-1)P_{k-2}(t)) / k
        P[:, k] = ((2.0 * k - 1.0) * t * P[:, k - 1] - (k - 1.0) * P[:, k - 2]) / k
    
    # Linear combination
    return P @ params
# EVOLVE-BLOCK-END
