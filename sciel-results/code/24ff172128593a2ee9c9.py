# EVOLVE-BLOCK-START
import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = _START_TIME + 600.0
_SAFETY_MARGIN = 5.0  # seconds to spare before the hard limit


def _area_from_positions(x):
    """
    Compute the area of the union of the 64 triangles using the prescribed quadrature.
    x: array of shape (64,)
    Returns: float area
    """
    n = 64
    # Heights: y_m = (m + 0.5) / 20000, m = 0..19999
    m = np.arange(20000, dtype=np.float64)
    y = (m + 0.5) / 20000.0  # shape (20000,)

    # a_i = (i + 1) / 64 for i = 0..63
    a = (np.arange(n, dtype=np.float64) + 1.0) / n  # shape (64,)

    # L_i(y) = x_i + a_i * y
    # We need to sort L_i at each height y_m.
    # L_i(y_m) = x_i + a_i * y_m
    # This is a linear function in y_m. The ordering can change as y changes.
    # However, since n=64 and 20000 heights, we can just compute all L values and sort them.
    # L shape: (20000, 64)
    L = x[None, :] + a[None, :] * y[:, None]  # (20000, 64)

    # Sort L along axis 1
    L_sorted = np.sort(L, axis=1)  # (20000, 64)

    # w = (1 - y) / 64
    w = (1.0 - y) / n  # (20000,)

    # Union length at each height:
    # union_len = w + sum_{j=0}^{62} min(L_sorted[j+1] - L_sorted[j], w)
    diffs = np.diff(L_sorted, axis=1)  # (20000, 63)
    capped_diffs = np.minimum(diffs, w[:, None])  # (20000, 63)
    union_len = w + capped_diffs.sum(axis=1)  # (20000,)

    area = np.mean(union_len)
    return float(area)


def _keich_baseline():
    """
    Classical Keich construction for n a power of 2.
    """
    n = 64
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    positions = []
    for i in range(n):
        # Binary representation of i, padded to k bits
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        positions.append(np.sum((1 - j) * eps * (2.0 ** -j) / k))
    return np.array(positions, dtype=np.float64)


def _area_proxy(x):
    """
    A faster proxy for the area, using fewer quadrature points.
    Used during optimization for speed.
    """
    n = 64
    # Use 2000 points instead of 20000
    m = np.arange(2000, dtype=np.float64)
    y = (m + 0.5) / 20000.0

    a = (np.arange(n, dtype=np.float64) + 1.0) / n
    L = x[None, :] + a[None, :] * y[:, None]
    L_sorted = np.sort(L, axis=1)
    w = (1.0 - y) / n
    diffs = np.diff(L_sorted, axis=1)
    capped_diffs = np.minimum(diffs, w[:, None])
    union_len = w + capped_diffs.sum(axis=1)
    return float(np.mean(union_len))


def _coordinate_descent(x, max_iters=500, step_sizes=None):
    """
    Coordinate descent with adaptive step sizes to minimize the area.
    """
    n = 64
    if step_sizes is None:
        step_sizes = [0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001]

    x = x.copy()
    current_area = _area_proxy(x)

    for step in step_sizes:
        improved = True
        while improved:
            improved = False
            for i in range(n):
                # Try moving x[i] by +/- step
                best_x = x[i]
                best_area = current_area

                for delta in [-step, step]:
                    x_new = x.copy()
                    x_new[i] = x[i] + delta
                    area_new = _area_proxy(x_new)
                    if area_new < best_area:
                        best_area = area_new
                        best_x = x_new[i]

                if best_x != x[i]:
                    x[i] = best_x
                    current_area = best_area
                    improved = True

            # Check deadline
            if time.time() > _DEADLINE - _SAFETY_MARGIN:
                return x

    return x


def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Two-dimensional Kakeya-needle area-minimization problem. For a fixed n, choose
    base positions x = (x_0, ..., x_{n-1}). For position index i (0-based) let
    a_i = (i + 1) / n and define the triangle R_i with vertices
        (x_i, 0), (x_i + 1/n, 0), (x_i + a_i, 1).
    Every triangle has base width 1/n on the line y = 0 and its apex on the line
    y = 1. The objective is to choose x so that the AREA of the union of the n
    triangles R_i is as small as possible. Here n = 64.

    Returns:
        positions: np.ndarray of shape (64,) with the base x-coordinates x_i.
    """
    # Start with the Keich baseline
    x = _keich_baseline()
    best_x = x.copy()
    best_area = _area_from_positions(x)

    # Coordinate descent to improve
    x_cd = _coordinate_descent(x)
    area_cd = _area_from_positions(x_cd)
    if area_cd < best_area:
        best_x = x_cd
        best_area = area_cd

    # Check deadline
    if time.time() > _DEADLINE - _SAFETY_MARGIN:
        return best_x

    # Try a few random perturbations + local search
    rng = np.random.default_rng(42)
    for _ in range(50):
        if time.time() > _DEADLINE - _SAFETY_MARGIN:
            break

        # Perturb the best solution
        x_pert = best_x + rng.normal(0, 0.01, size=64)
        x_pert = _coordinate_descent(x_pert, max_iters=100,
                                     step_sizes=[0.001, 0.0005, 0.0001])
        area_pert = _area_from_positions(x_pert)
        if area_pert < best_area:
            best_x = x_pert
            best_area = area_pert

    return best_x


# EVOLVE-BLOCK-END
