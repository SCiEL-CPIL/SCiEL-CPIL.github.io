"""Circle packing in perimeter-4 rectangle: 21 circles, maximize sum of radii."""
import numpy as np
from scipy.optimize import minimize
import time

N = 21


def _softplus(x, beta=100.0):
    """Numerically stable softplus."""
    xb = beta * x
    return np.where(xb > 0,
                    x + np.log1p(np.exp(-xb)) / beta,
                    np.log1p(np.exp(xb)) / beta)


def _softplus_grad(x, beta=100.0):
    """Gradient of softplus: sigmoid(beta*x)."""
    return 1.0 / (1.0 + np.exp(-np.clip(beta * x, -30, 30)))


def _objective_and_grad(x, w, h, beta=100.0, eq_weight=0.05):
    """
    Softplus-penalized objective with analytical gradients (vectorized).
    x: shape (3N,) = [x0, y0, r0, x1, y1, r1, ...]
    Minimizes: -sum(r) + boundary penalties + pairwise overlap penalties + equalization penalty.
    """
    c = x.reshape(-1, 3)
    xs = c[:, 0]
    ys = c[:, 1]
    rs = c[:, 2]

    obj = -np.sum(rs)
    grad = np.zeros_like(x)

    # Equalization pressure: penalize variance of radii to push toward symmetric packing
    if eq_weight > 0:
        r_mean = np.mean(rs)
        r_var = np.mean((rs - r_mean) ** 2)
        obj += eq_weight * r_var
        grad[2::3] += eq_weight * 2.0 * (rs - r_mean) / N

    # Boundary penalties
    v1 = rs - xs
    obj += np.sum(_softplus(v1, beta))
    g1 = _softplus_grad(v1, beta)
    grad[0::3] -= g1
    grad[2::3] += g1

    v2 = xs + rs - w
    obj += np.sum(_softplus(v2, beta))
    g2 = _softplus_grad(v2, beta)
    grad[0::3] += g2
    grad[2::3] += g2

    v3 = rs - ys
    obj += np.sum(_softplus(v3, beta))
    g3 = _softplus_grad(v3, beta)
    grad[1::3] -= g3
    grad[2::3] += g3

    v4 = ys + rs - h
    obj += np.sum(_softplus(v4, beta))
    g4 = _softplus_grad(v4, beta)
    grad[1::3] += g4
    grad[2::3] += g4

    v5 = -rs
    obj += np.sum(_softplus(v5, beta))
    g5 = _softplus_grad(v5, beta)
    grad[2::3] -= g5

    # Pairwise overlap penalties (vectorized per i)
    for i in range(N):
        dx = xs[i] - xs[i + 1:]
        dy = ys[i] - ys[i + 1:]
        d = np.sqrt(dx * dx + dy * dy + 1e-12)
        v = rs[i] + rs[i + 1:] - d
        obj += np.sum(_softplus(v, beta))
        g = _softplus_grad(v, beta)

        grad[3 * i] += -np.sum(g * dx / d)
        grad[3 * i + 1] += -np.sum(g * dy / d)
        grad[3 * i + 2] += np.sum(g)

        idx = np.arange(i + 1, N)
        grad[3 * idx] += g * dx / d
        grad[3 * idx + 1] += g * dy / d
        grad[3 * idx + 2] += g

    # Gradient of -sum(r)
    grad[2::3] += -1.0

    return obj, grad


def _check_valid(circles, w, h, tol=1e-6):
    """Check if circles configuration is valid."""
    for i in range(N):
        xi, yi, ri = circles[i]
        if ri < 1e-9:
            return False
        if xi - ri < -tol or xi + ri > w + tol or yi - ri < -tol or yi + ri > h + tol:
            return False
    for i in range(N):
        for j in range(i + 1, N):
            xi, yi, ri = circles[i]
            xj, yj, rj = circles[j]
            dist_sq = (xi - xj) ** 2 + (yi - yj) ** 2
            min_dist = ri + rj
            if dist_sq < (min_dist - tol) ** 2:
                return False
    return True


def _hex_init(w, h, seed=42):
    """Hexagonal lattice init over several row configurations."""
    rng = np.random.RandomState(seed)
    row_configs = [
        [3, 4, 4, 4, 3, 3],
        [4, 4, 4, 4, 3, 2],
        [2, 4, 4, 4, 4, 3],
        [3, 4, 4, 4, 4, 2],
        [4, 3, 4, 4, 4, 2],
        [2, 3, 4, 4, 4, 4],
        [3, 3, 4, 4, 4, 3],
        [4, 4, 3, 4, 4, 2],
        [2, 4, 3, 4, 4, 4],
        [3, 4, 3, 4, 4, 3],
        [5, 4, 4, 4, 4],
        [4, 5, 4, 4, 4],
        [3, 5, 4, 4, 5],
        [4, 4, 5, 4, 4],
        [2, 5, 4, 5, 5],
        [5, 5, 5, 5, 1],
        [1, 5, 5, 5, 5],
        [3, 4, 5, 4, 5],
        [5, 4, 5, 4, 3],
        [4, 4, 4, 4, 4, 1],
        [1, 4, 4, 4, 4, 4],
    ]
    best = None
    best_val = -np.inf

    for row_counts in row_configs:
        if sum(row_counts) != N:
            continue
        n_rows = len(row_counts)
        max_cols = max(row_counts)
        est_r_h = w / (max_cols + 0.5)
        est_r_v = h / (n_rows * np.sqrt(3) / 2 + 1.0)
        r_est = min(est_r_h, est_r_v) * 0.95
        if r_est < 0.01:
            continue

        circles = []
        for row_idx, n_cols in enumerate(row_counts):
            y = (row_idx + 0.5) * r_est * np.sqrt(3)
            x_offset = r_est if row_idx % 2 == 1 else 0.0
            row_width = n_cols * 2 * r_est
            center_x = w / 2
            for col_idx in range(n_cols):
                x = center_x - row_width / 2 + x_offset + (col_idx + 0.5) * 2 * r_est
                circles.append([x, y, r_est])

        circles = np.array(circles)
        if _check_valid(circles, w, h):
            val = float(np.sum(circles[:, 2]))
            if val > best_val:
                best_val = val
                best = circles.copy()

    if best is None:
        # Fallback grid
        a = int(np.ceil(np.sqrt(N * w / h)))
        b = int(np.ceil(N / a))
        while a * b < N:
            b += 1
        circles = []
        for i in range(N):
            col = i % a
            row = i // a
            r = min(w / (a + 0.5), h / (b + 0.5)) * 0.9
            x = (col + 0.5) * w / a
            y = (row + 0.5) * h / b
            circles.append([x, y, r])
        best = np.array(circles)

    return best


def _grid_init(w, h, seed=42):
    """Simple grid initialization."""
    rng = np.random.RandomState(seed)
    a = int(np.ceil(np.sqrt(N * w / h)))
    b = int(np.ceil(N / a))
    while a * b < N:
        b += 1
    circles = []
    for i in range(N):
        col = i % a
        row = i // a
        r = min(w / (a + 0.5), h / (b + 0.5)) * 0.9
        x = (col + 0.5) * w / a
        y = (row + 0.5) * h / b
        circles.append([x, y, r])
    return np.array(circles)


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin

    best_value = -np.inf
    best_result = None

    def check_and_update(circles, w, h):
        nonlocal best_value, best_result
        if time.time() > deadline:
            return
        if _check_valid(circles, w, h):
            val = float(np.sum(circles[:, 2]))
            if val > best_value:
                best_value = val
                best_result = circles.copy()

    # Try different aspect ratios
    aspect_ratios = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]

    for ar in aspect_ratios:
        if time.time() > deadline - 10:
            break

        # W + H = 2, W/H = ar => W = 2*ar/(1+ar), H = 2/(1+ar)
        w = 2.0 * ar / (1.0 + ar)
        h = 2.0 / (1.0 + ar)

        # Generate initial configurations
        init_configs = []

        # Hexagonal
        hex_circles = _hex_init(w, h, seed=int(ar * 100))
        init_configs.append(hex_circles)

        # Grid
        grid_circles = _grid_init(w, h, seed=int(ar * 100) + 1)
        init_configs.append(grid_circles)

        # Random
        rng = np.random.RandomState(int(ar * 100) + 2)
        for _ in range(3):
            r_est = min(w / 6, h / 4) * 0.8
            circles = np.array([
                [rng.uniform(r_est, w - r_est), rng.uniform(r_est, h - r_est), r_est * rng.uniform(0.8, 1.2)]
                for _ in range(N)
            ])
            init_configs.append(circles)

        for init_circles in init_configs:
            if time.time() > deadline - 10:
                break

            # Multi-phase optimization with increasing beta
            current = init_circles.copy()

            # Phase 1: Low beta, higher equalization
            x_flat = current.flatten()
            try:
                res = minimize(
                    lambda x: _objective_and_grad(x, w, h, beta=20.0, eq_weight=0.1)[0],
                    x_flat,
                    jac=lambda x: _objective_and_grad(x, w, h, beta=20.0, eq_weight=0.1)[1],
                    method='L-BFGS-B',
                    bounds=[(0, w), (0, h), (0.001, 1.0)] * N,
                    options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-6}
                )
                if res.success or res.fun < 1e-3:
                    current = res.x.reshape(-1, 3)
            except Exception:
                pass

            check_and_update(current, w, h)

            # Phase 2: Medium beta
            if time.time() > deadline - 10:
                break
            x_flat = current.flatten()
            try:
                res = minimize(
                    lambda x: _objective_and_grad(x, w, h, beta=50.0, eq_weight=0.05)[0],
                    x_flat,
                    jac=lambda x: _objective_and_grad(x, w, h, beta=50.0, eq_weight=0.05)[1],
                    method='L-BFGS-B',
                    bounds=[(0, w), (0, h), (0.001, 1.0)] * N,
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if res.success or res.fun < 1e-4:
                    current = res.x.reshape(-1, 3)
            except Exception:
                pass

            check_and_update(current, w, h)

            # Phase 3: High beta, low equalization
            if time.time() > deadline - 10:
                break
            x_flat = current.flatten()
            try:
                res = minimize(
                    lambda x: _objective_and_grad(x, w, h, beta=100.0, eq_weight=0.01)[0],
                    x_flat,
                    jac=lambda x: _objective_and_grad(x, w, h, beta=100.0, eq_weight=0.01)[1],
                    method='L-BFGS-B',
                    bounds=[(0, w), (0, h), (0.001, 1.0)] * N,
                    options={'maxiter': 800, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                if res.success or res.fun < 1e-5:
                    current = res.x.reshape(-1, 3)
            except Exception:
                pass

            check_and_update(current, w, h)

            # Phase 4: Very high beta, no equalization
            if time.time() > deadline - 10:
                break
            x_flat = current.flatten()
            try:
                res = minimize(
                    lambda x: _objective_and_grad(x, w, h, beta=200.0, eq_weight=0.0)[0],
                    x_flat,
                    jac=lambda x: _objective_and_grad(x, w, h, beta=200.0, eq_weight=0.0)[1],
                    method='L-BFGS-B',
                    bounds=[(0, w), (0, h), (0.001, 1.0)] * N,
                    options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12}
                )
                if res.success or res.fun < 1e-6:
                    current = res.x.reshape(-1, 3)
            except Exception:
                pass

            check_and_update(current, w, h)

    # Fallback to a simple valid configuration
    if best_result is None:
        w = 1.0
        h = 1.0
        a = 5
        b = 5
        circles = []
        for i in range(N):
            col = i % a
            row = i // a
            r = min(w / (a + 0.5), h / (b + 0.5)) * 0.9
            x = (col + 0.5) * w / a
            y = (row + 0.5) * h / b
            circles.append([x, y, r])
        best_result = np.array(circles)

    # Final validation
    if not _check_valid(best_result, w, h, tol=1e-4):
        # Try to fix by clamping
        for i in range(N):
            xi, yi, ri = best_result[i]
            ri = max(ri, 0.001)
            xi = np.clip(xi, ri, w - ri) if w > 2 * ri else w / 2
            yi = np.clip(yi, ri, h - ri) if h > 2 * ri else h / 2
            best_result[i] = [xi, yi, ri]

    return best_result
