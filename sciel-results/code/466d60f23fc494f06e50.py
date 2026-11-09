"""Symbolic regression: fit 10 coefficients over a well-conditioned smooth basis."""
# EVOLVE-BLOCK-START
import numpy as np


def _basis(x, y):
    """Return an (N, 10) matrix of smooth, bounded basis functions on [-1,1]^2.

    Design:
      - Bounded in magnitude (|t|<=1) so features are well-conditioned.
      - Linearly independent, smooth, and able to represent a wide family of
        continuous targets (polynomial + low-frequency trigonometric + radial).
      - Normalized so each column has comparable scale (helps BFGS from ones).
    """
    n = x.shape[0]
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Bounded polynomial-like features (Chebyshev on [-1,1], already bounded)
    c1x = x
    c2x = 2 * x * x - 1
    c1y = y
    c2y = 2 * y * y - 1

    f = np.empty((n, 10), dtype=float)
    f[:, 0] = 1.0
    f[:, 1] = c1x
    f[:, 2] = c1y
    f[:, 3] = c2x
    f[:, 4] = c2y
    f[:, 5] = c1x * c1y
    f[:, 6] = c2x * c2y
    f[:, 7] = np.sin(np.pi * c1x)
    f[:, 8] = np.sin(np.pi * c1y)
    f[:, 9] = np.cos(np.pi * (c1x + c1y))

    # Column-wise scale normalization (robust): divide by RMS, guard tiny cols.
    scale = np.sqrt(np.mean(f * f, axis=0))
    scale = np.where(scale > 1e-8, scale, 1.0)
    f = f / scale
    return f


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Closed-form linear combination of the basis; params are the 10 coefficients."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    p = np.asarray(params, dtype=float)
    B = _basis(x, y)
    return B @ p
# EVOLVE-BLOCK-END
