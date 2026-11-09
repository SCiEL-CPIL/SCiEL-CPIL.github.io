# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _compute_area_and_grad(x, n=64, M=20000):
    """Compute area and gradient of union area using vectorized operations."""
    a = (np.arange(n) + 1) / n
    w0 = 1.0 / n
    y = (np.arange(M) + 0.5) / M

    L = x[:, None] + a[:, None] * y[None, :]
    w = (1.0 - y[None, :]) * w0

    idx = np.argsort(L, axis=0)
    L_sorted = np.take_along_axis(L, idx, axis=0)
    gaps = np.diff(L_sorted, axis=0)

    union_lengths = w + np.minimum(gaps, w[None, :]).sum(axis=0)
    area = np.mean(union_lengths)

    active = gaps < w[None, :]
    left_idx = idx[:-1, :].ravel()
    right_idx = idx[1:, :].ravel()
    active_flat = active.ravel()

    left_counts = np.bincount(left_idx, weights=active_flat, minlength=n)
    right_counts = np.bincount(right_idx, weights=active_flat, minlength=n)

    grad_x = (left_counts - right_counts) / M

    return area, grad_x


def _compute_area_fast(x, n=64, M=20000):
    """Compute only the area (no gradient) for evaluation."""
    a = (np.arange(n) + 1) / n
    w0 = 1.0 / n
    y = (np.arange(M) + 0.5) / M

    L = x[:, None] + a[:, None] * y[None, :]
    w = (1.0 - y[None, :]) * w0

    idx = np.argsort(L, axis=0)
    L_sorted = np.take_along_axis(L, idx, axis=0)
    gaps = np.diff(L_sorted, axis=0)

    union_lengths = w + np.minimum(gaps, w[None, :]).sum(axis=0)
    area = np.mean(union_lengths)
    return area


def _compute_area_softmin_grad(x, n=64, M=20000, beta=50.0):
    """
    Compute area and gradient using soft-min surrogate.
    """
    a = (np.arange(n) + 1) / n
    w0 = 1.0 / n
    y = (np.arange(M) + 0.5) / M

    L = x[:, None] + a[:, None] * y[None, :]
    w = (1.0 - y[None, :]) * w0

    idx = np.argsort(L, axis=0)
    L_sorted = np.take_along_axis(L, idx, axis=0)
    gaps = np.diff(L_sorted, axis=0)

    diff = gaps - w[None, :]
    abs_diff = np.abs(diff)
    soft_correction = beta * np.log1p(np.exp(-abs_diff / beta))
    soft_min_vals = np.minimum(gaps, w[None, :]) - soft_correction

    union_lengths = w + soft_min_vals.sum(axis=0)
    area = np.mean(union_lengths)

    sigmoid_vals = 1.0 / (1.0 + np.exp(-diff / beta))

    left_idx = idx[:-1, :].ravel()
    right_idx = idx[1:, :].ravel()
    sig_flat = sigmoid_vals.ravel()

    left_counts = np.bincount(left_idx, weights=sig_flat, minlength=n)
    right_counts = np.bincount(right_idx, weights=sig_flat, minlength=n)

    grad_x = (left_counts - right_counts) / M

    return area, grad_x


def _keich_initial(n=64):
    """Generate the Keich construction initial point."""
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    positions = []
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        positions.append(np.sum((1 - j) * eps * (2.0 ** -j) / k))
    return np.array(positions, dtype=np.float64)


def _subgradient_polish(x, n=64, M=20000, deadline=None, max_iter=200):
    """
    Subgradient-based polish with adaptive step sizes for the nonsmooth objective.
    """
    x = x.copy()
    best_area = _compute_area_fast(x, n, M)
    best_x = x.copy()
    
    a = (np.arange(n) + 1) / n
    w0 = 1.0 / n
    y = (np.arange(M) + 0.5) / M
    L_base = a[:, None] * y[None, :]
    w = (1.0 - y[None, :]) * w0

    step_size = 0.01
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline - 10:
            break

        area, grad = _compute_area_and_grad(x, n, M)
        if area < best_area - 1e-15:
            best_area = area
            best_x = x.copy()

        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-10:
            break

        x_new = x - step_size * grad / (grad_norm + 1e-15)
        new_area = _compute_area_fast(x_new, n, M)

        if new_area < area:
            x = x_new
            step_size = min(step_size * 1.1, 0.05)
        else:
            step_size *= 0.8
            if step_size < 1e-7:
                break

    return best_x, best_area


def _coordinate_descent_polish(x, n=64, M=20000, max_rounds=30, deadline=None):
    """
    Coordinate descent polish: for each coordinate, use ternary search to find
    the optimal position that minimizes area.
    """
    x = x.copy()
    best_area = _compute_area_fast(x, n, M)
    a = (np.arange(n) + 1) / n
    w0 = 1.0 / n
    y = (np.arange(M) + 0.5) / M

    L_base = a[:, None] * y[None, :]
    w = (1.0 - y[None, :]) * w0

    def _area_with_xi(xi_val, xi_idx):
        L = x[:, None] + L_base
        L[xi_idx, :] = xi_val + L_base[xi_idx, :]
        idx = np.argsort(L, axis=0)
        L_sorted = np.take_along_axis(L, idx, axis=0)
        gaps = np.diff(L_sorted, axis=0)
        union_lengths = w + np.minimum(gaps, w[None, :]).sum(axis=0)
        return np.mean(union_lengths)

    for round_idx in range(max_rounds):
        if deadline is not None and time.time() > deadline - 10:
            break

        improved = False
        for i in range(n):
            if deadline is not None and time.time() > deadline - 10:
                break

            current_xi = x[i]
            best_xi = current_xi
            best_local_area = best_area

            # Ternary search over a range around current position
            range_size = 0.02 * (0.9 ** round_idx) + 0.001
            lo = current_xi - range_size
            hi = current_xi + range_size

            for _ in range(25):
                m1 = lo + (hi - lo) / 3
                m2 = hi - (hi - lo) / 3
                f1 = _area_with_xi(m1, i)
                f2 = _area_with_xi(m2, i)

                if f1 < f2:
                    hi = m2
                else:
                    lo = m1

                if f1 < best_local_area - 1e-15:
                    best_local_area = f1
                    best_xi = m1
                if f2 < best_local_area - 1e-15:
                    best_local_area = f2
                    best_xi = m2

            if best_xi != current_xi:
                x[i] = best_xi
                best_area = best_local_area
                improved = True

        if not improved:
            break

    return x, best_area


def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Two-dimensional Kakeya-needle area-minimization problem for n=64.
    
    Uses multi-restart optimization with:
    1. L-BFGS-B with exact nonsmooth gradient
    2. Soft-min smooth surrogate with annealed beta for basin escape
    3. Subgradient polish
    4. Coordinate descent polish
    """
    start_time = time.time()
    deadline = start_time + 620  # 620 second wall-clock limit (10s safety margin)

    n = 64
    M = 20000

    x_keich = _keich_initial(n)
    best_x = x_keich.copy()
    best_area = _compute_area_fast(x_keich, n, M)

    rng = np.random.default_rng(12345)

    # Phase 1: Multi-start L-BFGS-B with exact gradient
    inits = [x_keich.copy()]
    for trial in range(8):
        scale = 0.01 * (trial + 1)
        inits.append(x_keich + rng.normal(0, scale, size=n))

    for x0 in inits:
        elapsed = time.time() - start_time
        if elapsed > deadline - 60:
            break

        try:
            def fun_and_jac(x_eval):
                area, grad = _compute_area_and_grad(x_eval, n, M)
                return area, grad

            result = minimize(
                fun=fun_and_jac,
                x0=x0,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )

            area_full = _compute_area_fast(result.x, n, M)
            if area_full < best_area:
                best_area = area_full
                best_x = result.x.copy()
        except Exception:
            pass

    # Phase 2: Soft-min annealing - progressively sharpen beta
    for beta in [200.0, 100.0, 50.0, 25.0, 12.0, 6.0, 3.0]:
        elapsed = time.time() - start_time
        if elapsed > deadline - 40:
            break

        try:
            def fun_and_jac_soft(x_eval, b=beta):
                area, grad = _compute_area_softmin_grad(x_eval, n, M, b)
                return area, grad

            result_soft = minimize(
                fun=fun_and_jac_soft,
                x0=best_x,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 150, 'ftol': 1e-10, 'gtol': 1e-6}
            )

            area_exact = _compute_area_fast(result_soft.x, n, M)
            if area_exact < best_area:
                best_area = area_exact
                best_x = result_soft.x.copy()
        except Exception:
            pass

    # Phase 3: Subgradient polish
    elapsed = time.time() - start_time
    if elapsed < deadline - 30:
        try:
            polished_x, polished_area = _subgradient_polish(
                best_x, n, M, deadline=deadline, max_iter=300
            )
            if polished_area < best_area:
                best_area = polished_area
                best_x = polished_x.copy()
        except Exception:
            pass

    # Phase 4: Coordinate descent polish
    elapsed = time.time() - start_time
    if elapsed < deadline - 15:
        try:
            polished_x, polished_area = _coordinate_descent_polish(
                best_x, n, M, max_rounds=20, deadline=deadline
            )
            if polished_area < best_area:
                best_area = polished_area
                best_x = polished_x.copy()
        except Exception:
            pass

    # Final L-BFGS-B polish
    elapsed = time.time() - start_time
    if elapsed < deadline - 5:
        try:
            def fun_and_jac2(x_eval):
                area, grad = _compute_area_and_grad(x_eval, n, M)
                return area, grad

            result = minimize(
                fun=fun_and_jac2,
                x0=best_x,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            area_full = _compute_area_fast(result.x, n, M)
            if area_full < best_area:
                best_area = area_full
                best_x = result.x.copy()
        except Exception:
            pass

    if not np.all(np.isfinite(best_x)):
        best_x = x_keich

    return best_x.astype(np.float64)


# EVOLVE-BLOCK-END
