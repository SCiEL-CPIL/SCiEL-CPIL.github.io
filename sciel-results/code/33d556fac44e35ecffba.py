import numpy as np
import time

# Product/interaction basis with 10 terms
# Structurally different from univariate Chebyshev: captures multiplicative interactions
# The evaluator fits params via BFGS starting from all ones.

def _basis(x, y, z):
    """Return a (N, 10) matrix of product/interaction basis functions."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    n = x.shape[0]

    # 10 product/interaction terms:
    # t0: 1 (constant)
    # t1: x
    # t2: y
    # t3: z
    # t4: x*y
    # t5: x*z
    # t6: y*z
    # t7: x*y*z
    # t8: x^2
    # t9: y^2

    t0 = np.ones(n, dtype=np.float64)
    t1 = x
    t2 = y
    t3 = z
    t4 = x * y
    t5 = x * z
    t6 = y * z
    t7 = x * y * z
    t8 = x * x
    t9 = y * y

    B = np.column_stack([t0, t1, t2, t3, t4, t5, t6, t7, t8, t9])

    # Ensure finite
    B = np.nan_to_num(B, nan=0.0, posinf=0.0, neginf=0.0)
    return B

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)

    # Guard: ensure params has length 10
    if p.shape[0] != 10:
        p = np.resize(p, 10)

    B = _basis(x, y, z)

    # Linear combination: B @ p
    result = B @ p

    # Ensure finite
    if not np.all(np.isfinite(result)):
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

    return result
