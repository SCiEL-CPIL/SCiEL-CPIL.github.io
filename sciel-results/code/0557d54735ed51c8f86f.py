"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    RBF + linear + quadratic basis with 10 parameters.
    Uses 5 Gaussian RBFs at fixed centers with shared width,
    plus linear and quadratic terms for flexibility.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Fixed RBF centers
    centers = np.array([
        [0.0, 0.0],
        [0.5, 0.5],
        [-0.5, -0.5],
        [0.5, -0.5],
        [-0.5, 0.5]
    ])

    # Compute RBF terms
    # x, y are 1D arrays of same length
    dx = x[:, np.newaxis] - centers[:, 0]
    dy = y[:, np.newaxis] - centers[:, 1]
    dist2 = dx ** 2 + dy ** 2

    # Use a softplus-like transform to keep width positive
    # width = 1 + p3, starting at 2 when p3=1
    width = 1.0 + p3
    rbf = np.exp(-width * dist2)  # shape (n, 5)

    # Linear and quadratic terms
    x2 = x * x
    y2 = y * y
    xy = x * y

    # Combine: offset + linear + quadratic + RBF weighted sum
    result = (p0 + p1 * x + p2 * y + p4 * x2 + p5 * y2 + p6 * xy)
    result = result + rbf[:, 0] * p7 + rbf[:, 1] * p8 + rbf[:, 2] * p9

    # We only have 10 params, so we can't have all 5 RBF weights separate
    # Let's restructure: use p7, p8, p9 for 3 RBFs, and tie the other 2
    # Actually, let me recount: p0-p9 is 10 params
    # p0: offset, p1: x, p2: y, p3: width, p4: x2, p5: y2, p6: xy
    # p7, p8, p9: 3 RBF weights (centers 0, 1, 2)
    # Centers 3, 4 share weight with 1, 2 respectively (symmetry)
    result = result + rbf[:, 3] * p8 + rbf[:, 4] * p9

    return result
# EVOLVE-BLOCK-END
