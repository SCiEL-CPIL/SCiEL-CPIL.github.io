"""Symbolic regression: linear-in-params model with a rich fixed basis.
BFGS converges globally because the model is linear in params.
"""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    10-parameter linear model with a fixed, rich basis on [-2, 2]:
      p[0]: 1.0            (bias)
      p[1]: x              (linear)
      p[2]: x^2            (quadratic)
      p[3]: sin(pi*x)      (low-freq sine)
      p[4]: cos(pi*x)      (low-freq cosine)
      p[5]: sin(2*pi*x)    (mid-freq sine)
      p[6]: cos(2*pi*x)    (mid-freq cosine)
      p[7]: exp(-(x-1)^2)  (Gaussian bump at x=1)
      p[8]: exp(-(x+1)^2)  (Gaussian bump at x=-1)
      p[9]: exp(-x^2)      (Gaussian bump at x=0)
    """
    p = params
    pi = np.pi

    x2 = x * x

    # Polynomial terms
    poly = p[0] + p[1] * x + p[2] * x2

    # Trigonometric terms
    trig = (p[3] * np.sin(pi * x) +
            p[4] * np.cos(pi * x) +
            p[5] * np.sin(2.0 * pi * x) +
            p[6] * np.cos(2.0 * pi * x))

    # Gaussian bump terms (fixed centers at -1, 0, +1; fixed width 1.0)
    bumps = (p[7] * np.exp(-(x - 1.0) ** 2) +
             p[8] * np.exp(-(x + 1.0) ** 2) +
             p[9] * np.exp(-x2))

    return poly + trig + bumps
# EVOLVE-BLOCK-END
