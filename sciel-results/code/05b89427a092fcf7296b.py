# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_area_and_subgrad(x, n=64, M=20000):
    """
    Compute the union area and subgradient for the Kakeya needle problem.
    Returns (area, subgrad) where subgrad has shape (n,).
    """
    a = (np.arange(n) + 1) / n
    y = (np.arange(M) + 0.5) / M
    w = (1.0 - y) / n

    # L[m, i] = x[i] + a[i] * y[m]
    L = x[None, :] + a[None, :] * y[:, None]

    # Sort L along axis 1
    L_sorted = np.sort(L, axis=1)
    gaps = np.diff(L_sorted, axis=1)

    w_b = w[:, None]
    clipped = np.minimum(gaps, w_b)
    union_len = w + clipped.sum(axis=1)
    area = union_len.mean()

    # Subgradient computation
    active = (gaps < w_b).astype(np.float64)  # (M, n-1)

    # For sorted position p:
    #   sg_sorted[m, p] = 0.5 * (I[active[m, p-1]] - I[active[m, p]])
    # where active[m, -1] = 0 and active[m, n-1] = 0

    is_right = np.zeros((M, n), dtype=np.float64)
    is_right[:, 1:] = active

    is_left = np.zeros((M, n), dtype=np.float64)
    is_left[:, :n-1] = active

    sg_sorted = 0.5 * (is_right - is_left)  # (M, n)

    # Map back to original indices
    sorted_order = np.argsort(L, axis=1)
    sg = np.zeros_like(L)
    np.put_along_axis(sg, sorted_order, sg_sorted, axis=1)

    subgrad = sg.mean(axis=0)
    return area, subgrad


def _compute_area_fast(x, n=64, M=20000):
    """Compute only the area (faster, no subgradient)."""
    a = (np.arange(n) + 1) / n
    y = (np.arange(M) + 0.5) / M
    w = (1.0 - y) / n

    L = x[None, :] + a[None, :] * y[:, None]
    L_sorted = np.sort(L, axis=1)
    gaps = np.diff(L_sorted, axis=1)
    w_b = w[:, None]
    clipped = np.minimum(gaps, w_b)
    union_len = w + clipped.sum(axis=1)
    return union_len.mean()


def _keich_initial(n=64):
    """Generate the Keich construction initial positions."""
    k = int(np.log2(n))
    j_arr = np.arange(1, k + 1)
    x = np.zeros(n, dtype=np.float64)
    for i in range(n):
        bits = [(i >> (k - 1 - jj)) & 1 for jj in range(k)]
        eps = np.array(bits, dtype=np.float64)
        x[i] = np.sum((1 - j_arr) * eps * (2.0 ** -j_arr) / k)
    return x


def _subgradient_descent(x0, n=64, M=20000, max_steps=3000, lr_init=0.05, deadline=None):
    """Subgradient descent with diminishing step size."""
    x = x0.copy()
    best_x = x.copy()
    best_area, _ = _compute_area_and_subgrad(best_x, n, M)

    step = 0
    while step < max_steps:
        if deadline is not None and time.time() > deadline:
            break

        area, sg = _compute_area_and_subgrad(x, n, M)

        if area < best_area:
            best_area = area
            best_x = x.copy()

        # Diminishing learning rate
        lr = lr_init / (1.0 + 0.002 * step)

        x = x - lr * sg
        step += 1

    return best_x, best_area


def _coordinate_descent(x0, n=64, M=20000, max_rounds=100, scan_range=0.2, n_scan=21, deadline=None):
    """
    Coordinate descent: for each coordinate, scan a range of values to find the best.
    """
    x = x0.copy()
    best_area = _compute_area_fast(x, n, M)

    for rnd in range(max_rounds):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(n):
            if deadline is not None and time.time() > deadline:
                break
            # Scan around current value
            vals = np.linspace(x[i] - scan_range, x[i] + scan_range, n_scan)
            best_val = x[i]
            best_local_area = best_area
            for v in vals:
                x[i] = v
                area = _compute_area_fast(x, n, M)
                if area < best_local_area:
                    best_local_area = area
                    best_val = v
            x[i] = best_val
            if best_local_area < best_area - 1e-12:
                best_area = best_local_area
                improved = True
            # Shrink scan range if no improvement
            if not improved:
                scan_range *= 0.7

    return x, best_area


def _scipy_optimize(x0, n=64, M=20000, deadline=None):
    """Use scipy's L-BFGS-B on a smoothed proxy if available."""
    try:
        from scipy.optimize import minimize
    except ImportError:
        return x0, _compute_area_fast(x0, n, M)

    a = (np.arange(n) + 1) / n
    y = (np.arange(M) + 0.5) / M
    w = (1.0 - y) / n

    def objective(x):
        L = x[None, :] + a[None, :] * y[:, None]
        L_sorted = np.sort(L, axis=1)
        gaps = np.diff(L_sorted, axis=1)
        w_b = w[:, None]
        clipped = np.minimum(gaps, w_b)
        union_len = w + clipped.sum(axis=1)
        return union_len.mean()

    # Use a subset of points for speed during optimization
    M_opt = min(M, 2000)
    y_opt = (np.arange(M_opt) + 0.5) / M_opt
    w_opt = (1.0 - y_opt) / n

    def objective_fast(x):
        L = x[None, :] + a[None, :] * y_opt[:, None]
        L_sorted = np.sort(L, axis=1)
        gaps = np.diff(L_sorted, axis=1)
        w_b = w_opt[:, None]
        clipped = np.minimum(gaps, w_b)
        union_len = w_opt + clipped.sum(axis=1)
        return union_len.mean()

    result = minimize(objective_fast, x0, method='L-BFGS-B',
                      options={'maxiter': 500, 'maxfun': 5000})
    x_opt = result.x
    area_opt = _compute_area_fast(x_opt, n, M)
    return x_opt, area_opt


def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Two-dimensional Kakeya-needle area-minimization problem for n=64.
    Uses a multi-stage optimization: subgradient descent, coordinate descent,
    and scipy L-BFGS-B.
    """
    n = 64
    M = 20000

    start_time = time.time()
    deadline = start_time + 570  # 60s safety margin before 630s limit

    # Stage 1: Subgradient descent from Keich initial
    x_keich = _keich_initial(n)
    x1, area1 = _subgradient_descent(x_keich, n, M, max_steps=2000, lr_init=0.03, deadline=deadline)

    best_x = x1.copy()
    best_area = area1

    if time.time() > deadline:
        return best_x.astype(np.float64)

    # Stage 2: Coordinate descent refinement
    x2, area2 = _coordinate_descent(x1, n, M, max_rounds=50, scan_range=0.1, n_scan=11, deadline=deadline)
    if area2 < best_area:
        best_x = x2.copy()
        best_area = area2

    if time.time() > deadline:
        return best_x.astype(np.float64)

    # Stage 3: Scipy L-BFGS-B refinement
    x3, area3 = _scipy_optimize(x2, n, M, deadline=deadline)
    if area3 < best_area:
        best_x = x3.copy()
        best_area = area3

    if time.time() > deadline:
        return best_x.astype(np.float64)

    # Stage 4: Another round of subgradient descent from best
    x4, area4 = _subgradient_descent(best_x, n, M, max_steps=1000, lr_init=0.01, deadline=deadline)
    if area4 < best_area:
        best_x = x4.copy()
        best_area = area4

    # Final validation
    final_area = _compute_area_fast(best_x, n, M)
    if not np.isfinite(final_area) or final_area <= 0:
        # Fallback to Keich
        best_x = _keich_initial(n)

    return best_x.astype(np.float64)


# EVOLVE-BLOCK-END
