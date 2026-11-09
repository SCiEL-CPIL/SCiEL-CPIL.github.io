"""Symbolic regression using global piecewise linear hat-function basis."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Model: y = sum_{k=0}^{9} p_k * H_k(x)
    where H_k are piecewise linear "hat" basis functions with evenly spaced
    knots across [-2, 2]. The model is strictly linear in params, so BFGS
    converges to the global least-squares solution.

    We use 10 hat functions:
    - H_0: left boundary ramp (trapezoid from x=-2 to x=knot_1)
    - H_9: right boundary ramp (trapezoid from x=knot_8 to x=2)
    - H_k (k=1..8): standard hat centered at knot_k with support [knot_{k-1}, knot_{k+1}]

    Knots are evenly spaced: knot_k = -2 + 4*k/9 for k=0..9
    This gives 10 knots, and interior hats for k=1..8.
    """
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)

    n = len(x)
    if n == 0:
        return np.empty(0, dtype=np.float64)

    # 10 evenly spaced knots across [-2, 2]
    # knot_k = -2 + 4*k/9, k=0..9
    knots = -2.0 + 4.0 * np.arange(10) / 9.0

    result = np.zeros(n, dtype=np.float64)

    # H_0: left boundary hat
    # Support: [-2, knots[1]]
    # Rises from 0 at x=-2 to 1 at x=knots[1]
    k1 = knots[1]
    width_left = k1 - (-2.0)
    if width_left > 1e-12:
        # Linear ramp from (-2, 0) to (k1, 1)
        mask = (x >= -2.0) & (x <= k1)
        result[mask] += params[0] * ((x[mask] - (-2.0)) / width_left)

    # H_9: right boundary hat
    # Support: [knots[8], 2]
    # Rises from 0 at x=knots[8] to 1 at x=2
    k8 = knots[8]
    width_right = 2.0 - k8
    if width_right > 1e-12:
        mask = (x >= k8) & (x <= 2.0)
        result[mask] += params[9] * ((x[mask] - k8) / width_right)

    # H_k for k=1..8: standard hat functions
    # Centered at knots[k], support [knots[k-1], knots[k+1]]
    for k in range(1, 9):
        left = knots[k - 1]
        center = knots[k]
        right = knots[k + 1]

        # Left half: from (left, 0) to (center, 1)
        w_left = center - left
        if w_left > 1e-12:
            mask = (x >= left) & (x <= center)
            result[mask] += params[k] * ((x[mask] - left) / w_left)

        # Right half: from (center, 1) to (right, 0)
        w_right = right - center
        if w_right > 1e-12:
            mask = (x >= center) & (x <= right)
            result[mask] += params[k] * ((right - x[mask]) / w_right)

    # Ensure finite output
    return np.nan_to_num(result, nan=0.0, posinf=1e8, neginf=-1e8)
# EVOLVE-BLOCK-END
