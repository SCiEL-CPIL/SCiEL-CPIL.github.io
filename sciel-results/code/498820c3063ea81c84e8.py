import numpy as np
import time

# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Scaled monomial polynomial basis of degree 0..9.
    Basis functions: (x/2)^k for k=0..9, all bounded in [-1, 1].
    This provides a well-conditioned linear model for BFGS optimization.
    A 10th-degree polynomial can approximate any smooth function on [-2,2]
    with high accuracy, and the uniform scaling ensures the optimizer
    landscape is well-behaved.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    original_shape = x.shape
    x_flat = np.asarray(x, dtype=float).flatten()
    N = len(x_flat)

    if N == 0:
        return np.empty(original_shape)

    # Check time budget early
    if time.time() > deadline:
        # Fallback: return zero predictions (or constant)
        return np.zeros(original_shape)

    # Scale x to [-1, 1]
    t = x_flat / 2.0

    # Build basis matrix (N, 10): columns are t^0, t^1, ..., t^9
    B = np.empty((N, 10), dtype=float)
    B[:, 0] = 1.0
    for k in range(1, 10):
        B[:, k] = B[:, k-1] * t

    # Check time budget before final computation
    if time.time() > deadline:
        return np.zeros(original_shape)

    p = np.asarray(params, dtype=float).flatten()
    y = B @ p

    # Ensure finite output
    if not np.all(np.isfinite(y)):
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    return y.reshape(original_shape)
# EVOLVE-BLOCK-END
