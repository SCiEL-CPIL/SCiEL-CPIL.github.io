"""Symbolic regression with 3 inputs – centered/scaled monomial + interaction basis."""
import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression to recover an unknown function of 3 inputs.
    Uses a linear-in-params model with centered, scaled monomial and interaction basis.
    Inputs are mapped to [-1, 1] for well-conditioned optimization via BFGS.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)

    # Map inputs from [0.5, 2] to [-1, 1]
    # center = 1.25, half_range = 0.75
    u = (x - 1.25) / 0.75
    v = (y - 1.25) / 0.75
    w = (z - 1.25) / 0.75

    # 10 basis functions:
    # f0: constant
    # f1-f3: linear in each variable
    # f4-f6: pairwise products
    # f7: trilinear product
    # f8: sum of squares (norm-squared feature)
    # f9: product of squared deviations (captures interaction of magnitudes)
    f0 = np.ones_like(x)
    f1 = u
    f2 = v
    f3 = w
    f4 = u * v
    f5 = u * w
    f6 = v * w
    f7 = u * v * w
    f8 = u * u + v * v + w * w
    f9 = (u * u) * (v * v) + (v * v) * (w * w) + (w * w) * (u * u)

    pred = (params[0] * f0 +
            params[1] * f1 +
            params[2] * f2 +
            params[3] * f3 +
            params[4] * f4 +
            params[5] * f5 +
            params[6] * f6 +
            params[7] * f7 +
            params[8] * f8 +
            params[9] * f9)

    return pred
