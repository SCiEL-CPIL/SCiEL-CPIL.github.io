"""Symbolic regression with a rich, well-conditioned basis: polynomial + trig + tanh + Gaussians."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 135.0  # safety margin before 150s kill


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression using a stable, well-conditioned basis set.
    
    Basis (10 terms):
      p0 * 1          (constant)
      p1 * x          (linear)
      p2 * x^2        (quadratic)
      p3 * x^3        (cubic)
      p4 * sin(x)     (trig)
      p5 * cos(x)     (trig)
      p6 * tanh(x)    (saturation)
      p7 * exp(-((x - 0.5)/0.6)^2)   -- Gaussian bump, center 0.5, width 0.6
      p8 * exp(-((x + 0.5)/0.6)^2)   -- Gaussian bump, center -0.5, width 0.6
      p9 * sin(2*x)   (higher-frequency trig)
    """
    # Time guard
    if time.time() > _DEADLINE:
        return params[0] + params[1] * x

    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)

    if p.size < 10:
        p = np.concatenate([p, np.zeros(10 - p.size)])
    p = p[:10]

    p0, p1, p2, p3, p4 = p[0], p[1], p[2], p[3], p[4]
    p5, p6, p7, p8, p9 = p[5], p[6], p[7], p[8], p[9]

    # Polynomial terms
    x2 = x * x
    x3 = x2 * x

    # Trigonometric terms
    sin_x = np.sin(x)
    cos_x = np.cos(x)
    sin_2x = np.sin(2.0 * x)

    # Hyperbolic term
    tanh_x = np.tanh(x)

    # Gaussian bumps: fixed centers and widths for stability
    sigma = 0.6
    gauss1 = p7 * np.exp(-((x - 0.5) / sigma) ** 2)
    gauss2 = p8 * np.exp(-((x + 0.5) / sigma) ** 2)

    # Combine all terms
    y = (p0 * np.ones_like(x) +
         p1 * x +
         p2 * x2 +
         p3 * x3 +
         p4 * sin_x +
         p5 * cos_x +
         p6 * tanh_x +
         gauss1 +
         gauss2 +
         p9 * sin_2x)

    # Ensure finite output
    if not np.all(np.isfinite(y)):
        return np.zeros_like(x)

    return y
# EVOLVE-BLOCK-END
