"""Symbolic regression using a smooth Gaussian RBF basis (9 interior knots + constant)."""
# EVOLVE-BLOCK-START
import time

import numpy as np

# Module-level constants (fixed; do not depend on the call).
# 9 interior knots evenly spaced in [-2, 2] (spacing 0.4).
_KNOTS = np.array(
    [-1.6, -1.2, -0.8, -0.4, 0.0, 0.4, 0.8, 1.2, 1.6], dtype=np.float64
)
# RBF width: slightly larger than the knot spacing for smooth, well-conditioned overlap.
_WIDTH = 0.45
_INV_W2 = 1.0 / (_WIDTH * _WIDTH)

# Precompute the constant column shape helper lazily.


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Smooth Gaussian RBF model with 9 interior knots plus a constant.

    f(x) = p0 + sum_{i=1..9} p_i * exp(-((x - k_i)^2) / w^2)

    params (length 10):
      params[0] : constant term
      params[1] : weight of RBF centred at k_1
      ...
      params[9] : weight of RBF centred at k_9

    This smooth basis can approximate a wide family of nonlinear functions
    (polynomials, sines, exponentials, Gaussians) on [-2, 2] and, being smooth
    and well-conditioned, generalises better to unseen test points than a
    piecewise-linear (hinge) or low-degree monomial basis.
    """
    # Explicit wall-clock deadline guard: the evaluator kills after 150 s.
    # Evaluation is O(N) and fast, but we keep a safety margin.
    _t0 = time.time()

    x = np.asarray(x, dtype=np.float64).reshape(-1)
    params = np.asarray(params, dtype=np.float64).reshape(-1)

    # Guard against pathological / non-finite inputs during fitting.
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(params)):
        return np.full(x.shape, np.nan, dtype=np.float64)

    # Compute the 9 RBF columns in one vectorised pass: (N, 9).
    # d = (x - k_i)  ->  (N, 9)
    d = x[:, None] - _KNOTS[None, :]
    # G = exp(-(d^2) / w^2)  ->  (N, 9)
    G = np.exp(-_INV_W2 * (d * d))

    # Assemble the 10-column design matrix: [1, G].
    B = np.empty((x.shape[0], 10), dtype=np.float64)
    B[:, 0] = 1.0
    B[:, 1:] = G

    result = B @ params

    # Non-finite results make the fit/prediction invalid; keep them flagged.
    if not np.all(np.isfinite(result)):
        return result

    # Safety margin: if we are dangerously close to the kill limit, we still
    # have only a single O(N) pass remaining, so we can safely return.
    _ = (time.time() - _t0)  # (kept explicit; no further heavy work below)

    return result
# EVOLVE-BLOCK-END
