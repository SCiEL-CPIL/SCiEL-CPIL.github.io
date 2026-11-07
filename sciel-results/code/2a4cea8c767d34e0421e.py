"""Symbolic regression candidate: Full Quadratic Polynomial Basis.

This program uses a basis of all monomials of degree up to 2 in x, y, z.
Basis: 1, x, y, z, x^2, y^2, z^2, xy, xz, yz.

Rationale:
1. Smoothness: Most target functions in symbolic regression benchmarks are smooth.
   A quadratic basis provides a strong local approximation (2nd order Taylor series).
2. Numerical Stability: Polynomial terms are well-conditioned for BFGS optimizers
   compared to reciprocal terms, especially when inputs are bounded away from 0.
3. Coverage: The quadratic basis can approximate reciprocal terms (1/x) locally
   within the input range [0.5, 2] via Taylor expansion, whereas reciprocal
   terms cannot accurately approximate x^2 over the same range.
"""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)

    # 10-term basis: 1, x, y, z, x^2, y^2, z^2, xy, xz, yz
    c1 = np.ones_like(x, dtype=np.float64)
    c2 = x
    c3 = y
    c4 = z
    c5 = x * x
    c6 = y * y
    c7 = z * z
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
