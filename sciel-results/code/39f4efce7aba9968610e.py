"""Symbolic regression with 2 inputs: high-order interaction basis + numerical safeguards."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_T0 = time.time()
_DEADLINE = 145.0  # safety margin before the 150s kill


def _features(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return a (n, 10) feature matrix: polynomial + high-order interaction basis.
    
    Terms chosen to span:
      - Linear: x, y
      - Quadratic: x^2, y^2, xy
      - Cubic: x^3, y^3, x^2*y, x*y^2
      - Quintic interaction: xy*(x^2+y^2) = x^3*y + x*y^3
    
    This basis was found to score significantly better than alternatives
    that include a constant or radial (x^2+y^2) term, suggesting the
    hidden function has strong multiplicative/interaction structure
    rather than additive radial symmetry.
    """
    x2 = x * x
    y2 = y * y
    xy = x * y

    f0 = x
    f1 = y
    f2 = x2
    f3 = y2
    f4 = xy
    f5 = x * x2          # x^3
    f6 = y * y2          # y^3
    f7 = x2 * y          # x^2 * y
    f8 = x * y2          # x * y^2
    f9 = xy * (x2 + y2)  # x*y*(x^2+y^2) = x^3*y + x*y^3

    F = np.stack([f0, f1, f2, f3, f4, f5, f6, f7, f8, f9], axis=1)
    return F


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Compute predictions as linear combination of 10 basis features.
    
    The evaluator fits params via BFGS to minimize training MSE.
    This function must be fast and numerically stable.
    """
    # Guard against excessive runtime
    if time.time() > _DEADLINE:
        F = _features(x, y)
        pred = F @ params
        if not np.all(np.isfinite(pred)):
            pred = np.where(np.isfinite(pred), pred, 0.0)
        return pred

    F = _features(x, y)

    # Compute prediction
    pred = F @ params

    # Ensure finite output
    if not np.all(np.isfinite(pred)):
        pred = np.where(np.isfinite(pred), pred, 0.0)

    return pred
# EVOLVE-BLOCK-END
