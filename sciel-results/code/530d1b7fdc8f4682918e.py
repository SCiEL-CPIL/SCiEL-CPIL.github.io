# EVOLVE-BLOCK-START
"""Structural change: piecewise-linear bump basis + a handful of global shapes.

The previous design was a single global linear model over 10 smooth functions.
For many hidden targets (peaks, plateaus, asymmetric shapes) a global basis
underfits. Here we introduce localized bump functions (triangle hats) centered
at several points in [-3, 3]. The linear fit over these bumps can reconstruct
piecewise-linear and peaked targets far more faithfully than a small set of
global polynomials/trig terms, while the remaining global features preserve
ability to model smooth background and oscillations.

Basis (10 features):
  0: 1
  1: x
  2: x^2
  3: sin(x)
  4: cos(x)
  5: tanh(x)
  6: triangle hat at x=-1
  7: triangle hat at x=0
  8: triangle hat at x=1
  9: triangle hat at x=2
"""
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)

    # Global features
    ones = np.ones_like(x, dtype=np.float64)
    x2 = x * x
    sinx = np.sin(x)
    cosx = np.cos(x)
    tanhx = np.tanh(x)

    # Localized triangle (bump) features
    # Triangle hat: max(0, 1 - |x - c| / w) with half-width w=1
    def hat(c: float, w: float = 1.0) -> np.ndarray:
        return np.maximum(0.0, 1.0 - np.abs(x - c) / w)

    h_1 = hat(-1.0)
    h_0 = hat(0.0)
    h_1p = hat(1.0)
    h_2 = hat(2.0)

    # Assemble predictions via elementwise products with params
    pred = (
        params[0] * ones
        + params[1] * x
        + params[2] * x2
        + params[3] * sinx
        + params[4] * cosx
        + params[5] * tanhx
        + params[6] * h_1
        + params[7] * h_0
        + params[8] * h_1p
        + params[9] * h_2
    )

    return pred
# EVOLVE-BLOCK-END
