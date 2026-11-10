# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_START_TIME = time.time()
_DEADLINE = _START_TIME + 625.0
_SAFETY_MARGIN = 5.0


def _area_from_positions(x, num_points=20000):
    """
    Compute the area of the union of the 64 triangles using the prescribed quadrature.
    x: array of shape (64,)
    Returns: float area
    """
    n = 64
    m = np.arange(num_points, dtype=np.float64)
    y = (m + 0.5) / num_points

    a = (np.arange(n, dtype=np.float64) + 1.0) / n

    L = x[None, :] + a[None, :] * y[:, None]
    L_sorted = np.sort(L, axis=1)

    w = (1.0 - y) / n

    diffs = np.diff(L_sorted, axis=1)
    capped_diffs = np.minimum(diffs, w[:, None])
    union_len = w + capped_diffs.sum(axis=1)

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
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        positions.append(np.sum((1 - j) * eps * (2.0 ** -j) / k))
    return np.array(positions, dtype=np.float64)


def _area_proxy(x, num_points=2000):
    """
    A faster proxy for the area, using fewer quadrature points.
    Used during optimization for speed.
    """
    return _area_from_positions(x, num_points=num_points)


def _coordinate_descent_scan(x, max_passes=20, opt_points=2000):
    """
    Coordinate descent with adaptive 1D scan over each coordinate.
    Uses a two-stage scan: coarse over a shrinking range, then fine
    around the best point found. Range shrinks each pass for convergence.
    """
    n = 64
    x = x.copy()
    current_area = _area_proxy(x, num_points=opt_points)

    for _pass in range(max_passes):
        if time.time() > _DEADLINE - _SAFETY_MARGIN:
            break

        improved = False
        # Adaptive range: start wider, shrink as we converge
        base_range = 0.15 / (1.0 + 0.2 * _pass)

        for i in range(n):
            if time.time() > _DEADLINE - _SAFETY_MARGIN:
                break

            x_i_current = x[i]

            # Coarse scan
            coarse_n = 30
            coarse_range = np.linspace(x_i_current - base_range, x_i_current + base_range, coarse_n)

            best_xi = x_i_current
            best_local_area = current_area

            for xi in coarse_range:
                x_temp = x.copy()
                x_temp[i] = xi
                area = _area_proxy(x_temp, num_points=opt_points)
                if area < best_local_area - 1e-12:
                    best_local_area = area
                    best_xi = xi

            # Fine scan around the best point
            if best_xi != x_i_current:
                fine_range = base_range / coarse_n
                fine_n = 15
                fine_scan = np.linspace(best_xi - fine_range, best_xi + fine_range, fine_n)
                for xi in fine_scan:
                    x_temp = x.copy()
                    x_temp[i] = xi
                    area = _area_proxy(x_temp, num_points=opt_points)
                    if area < best_local_area - 1e-12:
                        best_local_area = area
                        best_xi = xi

            if best_xi != x_i_current:
                x[i] = best_xi
                current_area = best_local_area
                improved = True

        if not improved:
            break

    return x


def _soft_min_proxy(x, num_points=2000, beta=200.0):
    """
    Smooth surrogate for the union area using log-sum-exp to approximate
    the min in the overlap computation. This gives a differentiable objective
    suitable for gradient-based optimizers.
    
    The union length at height y is: w + sum_j min(L_{j+1} - L_j, w)
    We approximate min(a, b) via soft-min: -logaddexp(-a/beta, -b/beta) * beta
    """
    n = 64
    m = np.arange(num_points, dtype=np.float64)
    y = (m + 0.5) / num_points

    a = (np.arange(n, dtype=np.float64) + 1.0) / n

    L = x[None, :] + a[None, :] * y[:, None]
    L_sorted = np.sort(L, axis=1)

    w = (1.0 - y) / n  # shape (num_points,)

    diffs = np.diff(L_sorted, axis=1)  # shape (num_points, n-1)
    w_col = w[:, None]  # shape (num_points, 1)

    # soft_min(diffs, w) = -beta * logaddexp(-diffs/beta, -w/beta)
    # This approximates min(diffs, w) when beta is large
    soft = -beta * np.logaddexp(-diffs / beta, -w_col / beta)
    union_len = w + soft.sum(axis=1)

    return float(np.mean(union_len))


def _scipy_optimize(x0, num_points=1000):
    """
    Use scipy.optimize.minimize (L-BFGS-B) to refine the solution.
    Uses the nonsmooth area directly with numerical gradients.
    """
    def objective(x):
        return _area_from_positions(x, num_points=num_points)

    try:
        result = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            options={'maxiter': 200, 'gtol': 1e-9, 'ftol': 1e-12}
        )
        return result.x
    except Exception:
        return x0


def _scipy_soft_min_optimize(x0, num_points=2000, beta=200.0):
    """
    Use scipy.optimize.minimize with a smooth soft-min surrogate.
    This may escape local minima that trap the nonsmooth coordinate descent.
    """
    def objective(x):
        return _soft_min_proxy(x, num_points=num_points, beta=beta)

    def gradient(x):
        # Numerical gradient via central differences (step size ~1e-5)
        grad = np.zeros_like(x)
        eps = 1e-5
        f0 = objective(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += eps
            x_minus = x.copy()
            x_minus[i] -= eps
            grad[i] = (objective(x_plus) - objective(x_minus)) / (2 * eps)
        return grad

    try:
        result = minimize(
            objective,
            x0,
            jac=gradient,
            method='L-BFGS-B',
            options={'maxiter': 300, 'gtol': 1e-10, 'ftol': 1e-14}
        )
        return result.x
    except Exception:
        return x0


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

    # Phase 1: Coordinate descent with scan (fast, robust)
    # Use a moderate number of points for the proxy
    x_cd = _coordinate_descent_scan(x, max_passes=30, opt_points=2000)
    area_cd = _area_from_positions(x_cd)
    if area_cd < best_area:
        best_x = x_cd
        best_area = area_cd

    # Check deadline
    if time.time() > _DEADLINE - _SAFETY_MARGIN:
        return best_x

    # Phase 2: Try to further refine with scipy (nonsmooth)
    x_scipy = _scipy_optimize(best_x, num_points=500)
    area_scipy = _area_from_positions(x_scipy)
    if area_scipy < best_area:
        best_x = x_scipy
        best_area = area_scipy

    # Check deadline
    if time.time() > _DEADLINE - _SAFETY_MARGIN:
        return best_x

    # Phase 2b: Smooth surrogate optimization (soft-min)
    x_soft = _scipy_soft_min_optimize(best_x, num_points=2000, beta=200.0)
    area_soft = _area_from_positions(x_soft)
    if area_soft < best_area:
        best_x = x_soft
        best_area = area_soft

    # Check deadline
    if time.time() > _DEADLINE - _SAFETY_MARGIN:
        return best_x

    # Phase 3: Final fine-tuning with coordinate descent at higher resolution
    x_final = _coordinate_descent_scan(best_x, max_passes=40, opt_points=5000)
    area_final = _area_from_positions(x_final)
    if area_final < best_area:
        best_x = x_final
        best_area = area_final

    # Check deadline
    if time.time() > _DEADLINE - _SAFETY_MARGIN:
        return best_x

    # Phase 3b: Very fine coordinate descent with small range for polish
    x_polish = best_x.copy()
    current_polish_area = best_area
    for _p in range(30):
        if time.time() > _DEADLINE - _SAFETY_MARGIN:
            break
        improved = False
        for i in range(64):
            if time.time() > _DEADLINE - _SAFETY_MARGIN:
                break
            # Very small range for final polish
            scan = np.linspace(x_polish[i] - 0.005, x_polish[i] + 0.005, 20)
            best_xi = x_polish[i]
            best_a = current_polish_area
            for xi in scan:
                x_t = x_polish.copy()
                x_t[i] = xi
                a_t = _area_from_positions(x_t, num_points=5000)
                if a_t < best_a - 1e-13:
                    best_a = a_t
                    best_xi = xi
            if best_xi != x_polish[i]:
                x_polish[i] = best_xi
                current_polish_area = best_a
                improved = True
        if not improved:
            break
    if current_polish_area < best_area:
        best_x = x_polish
        best_area = current_polish_area

    # Phase 4: Random restarts + local search
    rng = np.random.default_rng(42)
    for _ in range(15):
        if time.time() > _DEADLINE - _SAFETY_MARGIN:
            break

        # Perturb the best solution
        x_pert = best_x + rng.normal(0, 0.01, size=64)
        x_pert = _coordinate_descent_scan(x_pert, max_passes=15, opt_points=3000)
        area_pert = _area_from_positions(x_pert)
        if area_pert < best_area:
            best_x = x_pert
            best_area = area_pert

    return best_x


# EVOLVE-BLOCK-END
