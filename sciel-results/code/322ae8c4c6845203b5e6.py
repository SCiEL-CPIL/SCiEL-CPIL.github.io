"""Symbolic regression with 2 inputs: mixed polynomial + trigonometric + rational basis."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_T0 = time.time()
_DEADLINE = 145.0


def _features(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Return a (n, 10) feature matrix.
    Strategy: Cover a broad set of smooth function families:
    - Constant
    - Linear (x, y)
    - Quadratic (x^2, y^2, xy)
    - Trigonometric (sin(x), cos(y))
    - Cubic (x^3)
    - Rational (1/(1+x^2+y^2))
    This diverse basis can approximate polynomials, oscillatory functions, and rational forms.
    """
    n = x.shape[0]

    f0 = np.ones(n)
    f1 = x
    f2 = y
    f3 = x * x
    f4 = y * y
    f5 = x * y
    f6 = np.sin(x)
    f7 = np.cos(y)
    f8 = x * x * x
    f9 = 1.0 / (1.0 + x * x + y * y)

    F = np.column_stack([f0, f1, f2, f3, f4, f5, f6, f7, f8, f9])
    return F


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Guard against excessive runtime
    if time.time() > _DEADLINE:
        F = _features(x, y)
        return F @ params

    F = _features(x, y)
    pred = F @ params

    # Ensure finite output
    if not np.all(np.isfinite(pred)):
        mask = np.isfinite(pred)
        pred[~mask] = 0.0

    return pred
# EVOLVE-BLOCK-END
