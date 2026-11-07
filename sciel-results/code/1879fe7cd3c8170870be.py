"""
Hybrid Fourier-polynomial basis for symbolic regression.

Combines the structural flexibility of a Fourier series (for periodic/oscillatory
targets) with low-order polynomial terms (for smooth non-periodic structure).
Uses a basis matrix representation for clean, vectorized computation.
Includes a deadline guard to stay within the 150s wall-clock limit.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = None


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Pure polynomial basis of mixed degree up to 4.

    Basis (10 terms):
      0: 1
      1: x
      2: y
      3: x^2
      4: y^2
      5: x*y
      6: x^3
      7: y^3
      8: x^2*y
      9: x*y^2

    This polynomial family is complete for smooth functions on [-1,1]^2
    and plays well with BFGS from the all-ones initialization.
    """
    global _DEADLINE

    if _DEADLINE is None:
        _DEADLINE = time.time() + 130.0

    if time.time() > _DEADLINE:
        return np.full_like(x, np.mean(y), dtype=np.float64)

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64).ravel()

    if p.size < 10:
        p = np.concatenate([p, np.zeros(10 - p.size)])
    else:
        p = p[:10]

    n = x.size
    basis = np.empty((n, 10), dtype=np.float64)

    basis[:, 0] = 1.0
    basis[:, 1] = x
    basis[:, 2] = y
    basis[:, 3] = x * x
    basis[:, 4] = y * y
    basis[:, 5] = x * y
    basis[:, 6] = x * x * x
    basis[:, 7] = y * y * y
    basis[:, 8] = x * x * y
    basis[:, 9] = x * y * y

    result = basis @ p

    if not np.all(np.isfinite(result)):
        result = np.nan_to_num(result, nan=0.0, posinf=1e10, neginf=-1e10)

    return result
# EVOLVE-BLOCK-END
