# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximate the hidden function using a small fixed-architecture neural
    network (2 hidden units, ReLU activations) parameterized by 10 free
    coefficients. The network is:

        z1 = w1[0]*x + w1[1]
        z2 = w1[2]*x + w1[3]
        a1 = ReLU(z1)
        a2 = ReLU(z2)
        y  = w2[0]*a1 + w2[1]*a2 + b

    where w1 (4 params), w2 (2 params), b (1 param) are the 7 non-constant
    coefficients, and the remaining 3 params modulate the overall scale and
    add a small regularized offset to keep the fit stable.

    This closed-form, differentiable expression is suitable for BFGS
    optimization.
    """
    _T0 = time.time()
    _DEADLINE = 140.0

    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n == 0:
        return np.array([])

    p = np.asarray(params, dtype=np.float64)
    # Ensure we have at least 10 parameters; pad with zeros if fewer.
    if p.size < 10:
        p = np.concatenate([p, np.zeros(10 - p.size)])
    p = p[:10]

    # Extract parameters
    w1_0, w1_1, w1_2, w1_3 = p[0], p[1], p[2], p[3]
    w2_0, w2_1 = p[4], p[5]
    b = p[6]
    scale = p[7]
    # Remaining 2 params used for a small smooth offset to help conditioning
    offset = 0.01 * (p[8] + p[9])

    # Forward pass through the small network
    z1 = w1_0 * x + w1_1
    z2 = w1_2 * x + w1_3
    a1 = np.maximum(z1, 0.0)
    a2 = np.maximum(z2, 0.0)

    y = w2_0 * a1 + w2_1 * a2 + b
    y = scale * y + offset

    # Finite-output safety net
    if not np.all(np.isfinite(y)):
        y = np.where(np.isfinite(y), y, 0.0)

    # Deadline guard
    if time.time() - _T0 > _DEADLINE:
        return y

    return y
# EVOLVE-BLOCK-END
