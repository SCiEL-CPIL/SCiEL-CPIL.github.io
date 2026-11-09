# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _union_area(x):
    """Compute the union area of 64 triangles given base positions x."""
    n = 64
    a = (np.arange(n) + 1) / n  # a_i = (i+1)/64
    y = (np.arange(20000) + 0.5) / 20000  # 20000 heights
    # L_i(y) = x_i + a_i * y, w(y) = (1-y)/64
    # Shape: (20000, 64)
    L = x[np.newaxis, :] + a[np.newaxis, :] * y[:, np.newaxis]
    w = (1.0 - y) / n  # shape (20000,)
    # Sort L along axis=1
    L_sorted = np.sort(L, axis=1)
    # Consecutive differences
    diffs = np.diff(L_sorted, axis=1)  # shape (20000, 63)
    # Union length = w + sum(min(diffs, w))
    union_len = w + np.sum(np.minimum(diffs, w[:, np.newaxis]), axis=1)
    return np.mean(union_len)


def _union_area_grad(x):
    """Compute union area and its gradient w.r.t. x."""
    n = 64
    a = (np.arange(n) + 1) / n
    y = (np.arange(20000) + 0.5) / 20000
    L = x[np.newaxis, :] + a[np.newaxis, :] * y[:, np.newaxis]
    w = (1.0 - y) / n

    # Sort L and track which original index each sorted position came from
    order = np.argsort(L, axis=1)  # (20000, 64)
    L_sorted = np.take_along_axis(L, order, axis=1)
    diffs = np.diff(L_sorted, axis=1)  # (20000, 63)

    # Which pairs are "active" (diff < w)?
    active = (diffs < w[:, np.newaxis]).astype(np.float64)  # (20000, 63)

    # Union length
    union_len = w + np.sum(active * diffs, axis=1)
    area = np.mean(union_len)

    # Gradient: d(area)/d(x_i)
    # area = (1/M) * sum_m [ w_m + sum_j active[m,j] * (L_sorted[m,j+1] - L_sorted[m,j]) ]
    # d L[m,i] / d x_i = a_i
    # For each m, the sorted positions correspond to original indices order[m,:].
    # d/dx_k of the union length at height m:
    #   For each active pair j at height m:
    #     L_sorted[m,j+1] - L_sorted[m,j]
    #     d/dx_k = a_{order[m,j+1]} * 1 (if k == order[m,j+1]) - a_{order[m,j]} * 1 (if k == order[m,j])
    #   So the gradient contribution is:
    #     For each active pair j: +a_{order[m,j+1]} at index order[m,j+1], -a_{order[m,j]} at index order[m,j]
    #   Multiplied by active[m,j] / M

    M = 20000
    grad = np.zeros(n, dtype=np.float64)

    # Vectorized: for each height m and each active pair j,
    # add a[order[m,j+1]] to grad[order[m,j+1]] and subtract a[order[m,j]] from grad[order[m,j]]
    # weighted by active[m,j] / M

    # order has shape (M, 64), diffs has shape (M, 63)
    # left indices: order[:, :-1] shape (M, 63)
    # right indices: order[:, 1:] shape (M, 63)
    left_idx = order[:, :-1]   # (M, 63)
    right_idx = order[:, 1:]   # (M, 63)

    # For each (m, j) where active[m,j] > 0:
    #   grad[right_idx[m,j]] += active[m,j] * a[right_idx[m,j]] / M
    #   grad[left_idx[m,j]]  -= active[m,j] * a[left_idx[m,j]] / M

    # Use np.add.at for accumulation
    weight = active / M  # (M, 63)

    # Flatten for efficient accumulation
    r_flat = right_idx.ravel()  # (M*63,)
    l_flat = left_idx.ravel()
    w_flat = weight.ravel()
    a_r = a[r_flat]  # a values at right indices
    a_l = a[l_flat]  # a values at left indices

    np.add.at(grad, r_flat, w_flat * a_r)
    np.add.at(grad, l_flat, -w_flat * a_l)

    return area, grad


def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Two-dimensional Kakeya-needle area-minimization problem. For a fixed n, choose
    base positions x = (x_0, ..., x_{n-1}). For position index i (0-based) let
    a_i = (i + 1) / n and define the triangle R_i with vertices
        (x_i, 0), (x_i + 1/n, 0), (x_i + a_i, 1).
    Every triangle has base width 1/n on the line y = 0 and its apex on the line
    y = 1. The objective is to choose x so that the AREA of the union of the n
    triangles R_i is as small as possible. Here n = 64.

    Uses gradient-based optimization starting from the Keich construction.

    Returns:
        positions: np.ndarray of shape (64,) with the base x-coordinates x_i.
    """
    n = 64
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    positions = []
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        positions.append(np.sum((1 - j) * eps * (2.0 ** -j) / k))
    x0 = np.array(positions, dtype=np.float64)

    # Deadline guard: aim to finish well before 630s
    t_start = time.time()
    deadline = t_start + 600.0

    best_x = x0.copy()
    best_area = _union_area(best_x)

    # Define objective and gradient for scipy
    def obj_func(x):
        return _union_area(x)

    def jac_func(x):
        return _union_area_grad(x)[1]

    # L-BFGS-B with gradient
    try:
        result = minimize(
            obj_func,
            x0,
            jac=jac_func,
            method='L-BFGS-B',
            options={'maxiter': 500, 'maxfun': 5000, 'gtol': 1e-8, 'ftol': 1e-12}
        )
        if result.x is not None and np.all(np.isfinite(result.x)):
            area_r = _union_area(result.x)
            if area_r < best_area:
                best_x = result.x.copy()
                best_area = area_r
    except Exception:
        pass

    # Check time; if we have budget, try a second run with different options
    elapsed = time.time() - t_start
    if elapsed < deadline - 120:
        try:
            # Try Nelder-Mead as a local polish (no gradient needed)
            result2 = minimize(
                obj_func,
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-14}
            )
            if result2.x is not None and np.all(np.isfinite(result2.x)):
                area_r2 = _union_area(result2.x)
                if area_r2 < best_area:
                    best_x = result2.x.copy()
                    best_area = area_r2
        except Exception:
            pass

    # Final check
    elapsed = time.time() - t_start
    if elapsed < deadline - 60:
        # One more L-BFGS-B polish
        try:
            result3 = minimize(
                obj_func,
                best_x,
                jac=jac_func,
                method='L-BFGS-B',
                options={'maxiter': 300, 'maxfun': 3000, 'gtol': 1e-10, 'ftol': 1e-14}
            )
            if result3.x is not None and np.all(np.isfinite(result3.x)):
                area_r3 = _union_area(result3.x)
                if area_r3 < best_area:
                    best_x = result3.x.copy()
                    best_area = area_r3
        except Exception:
            pass

    # Ensure valid output
    if not np.all(np.isfinite(best_x)):
        best_x = x0

    return best_x


# EVOLVE-BLOCK-END
