"""Symbolic regression with a blended polynomial + exponential basis."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis: low-degree monomials on scaled x, plus a couple of
    exponential terms to capture non-polynomial smooth behavior.

    10 parameters:
      p[0:6]  -> coefficients for 1, z, z^2, z^3, z^4, z^5  (z = x/3 in [-1,1])
      p[6:8]  -> coefficients for exp(2*z), exp(-2*z)
      p[8]    -> coefficient for exp(-4*(z-0.5)^2)
      p[9]    -> coefficient for exp(-4*(z+0.5)^2)
    """
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)

    # Safety: pad or truncate params to length 10
    if p.size < 10:
        p = np.pad(p, (0, 10 - p.size), constant_values=0.0)
    else:
        p = p[:10]

    # Scale to [-1, 1]
    z = x / 3.0

    # Polynomial part: degree 5
    z2 = z * z
    z3 = z2 * z
    z4 = z2 * z2
    z5 = z4 * z

    result = (p[0] + p[1] * z + p[2] * z2 +
              p[3] * z3 + p[4] * z4 + p[5] * z5)

    # Exponential terms for non-polynomial smooth features
    result += p[6] * np.exp(2.0 * z)
    result += p[7] * np.exp(-2.0 * z)

    # Localized Gaussian bumps (fixed centers) to capture local features
    result += p[8] * np.exp(-4.0 * (z - 0.5) ** 2)
    result += p[9] * np.exp(-4.0 * (z + 0.5) ** 2)

    return result
# EVOLVE-BLOCK-END
