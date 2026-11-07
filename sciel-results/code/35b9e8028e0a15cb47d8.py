"""Symbolic regression candidate: hybrid reciprocal-pairwise basis.

Combines the structural insight from the top-scoring inspiration (reciprocal
terms 1/x, 1/y, 1/z capture rational/hyperbolic behavior) with the pairwise
interaction terms (xy, xz, yz) that capture multiplicative structure.
The 10-term basis is: 1, x, y, z, 1/x, 1/y, 1/z, xy, xz, yz.

This representation is numerically stable for inputs in [0.5, 2] and allows
the BFGS fit to model functions with both linear trends, reciprocal
asymptotes, and pairwise interactions—structures that a pure polynomial basis
cannot capture efficiently.
"""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)

    # Guard against division by zero (inputs are in [0.5, 2] so safe, but guard anyway)
    x_safe = np.where(np.abs(x) < 1e-12, 1e-12, x)
    y_safe = np.where(np.abs(y) < 1e-12, 1e-12, y)
    z_safe = np.where(np.abs(z) < 1e-12, 1e-12, z)

    # 10-term basis: 1, x, y, z, 1/x, 1/y, 1/z, xy, xz, yz
    c1 = np.ones_like(x, dtype=np.float64)
    c2 = x
    c3 = y
    c4 = z
    c5 = 1.0 / x_safe
    c6 = 1.0 / y_safe
    c7 = 1.0 / z_safe
    c8 = x * y
    c9 = x * z
    c10 = y * z

    out = (
        p[0] * c1
        + p[1] * c2
        + p[2] * c3
        + p[3] * c4
        + p[4] * c5
        + p[5] * c6
        + p[6] * c7
        + p[7] * c8
        + p[8] * c9
        + p[9] * c10
    )
    return out
# EVOLVE-BLOCK-END
