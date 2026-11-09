"""Chebyshev basis with 10 coefficients, mapped to [0.5, 2]."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Map inputs from [0.5, 2] to [-1, 1]
    tx = (2.0 * x - 2.5) / 1.5
    ty = (2.0 * y - 2.5) / 1.5
    tz = (2.0 * z - 2.5) / 1.5

    # Chebyshev T1 and T2
    T1x = tx
    T1y = ty
    T1z = tz
    T2x = 2.0 * tx * tx - 1.0
    T2y = 2.0 * ty * ty - 1.0
    T2z = 2.0 * tz * tz - 1.0

    p = params

    return (
        p[0] +
        p[1] * T1x +
        p[2] * T1y +
        p[3] * T1z +
        p[4] * T1x * T1y +
        p[5] * T1x * T1z +
        p[6] * T1y * T1z +
        p[7] * T2x +
        p[8] * T2y +
        p[9] * T2z
    )
# EVOLVE-BLOCK-END
