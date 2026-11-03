"""Circle packing in perimeter-4 rectangle: gradient-based softplus penalty optimization
with vectorized objective, Delaunay-accelerated polish, multi-phase increasing-beta refinement,
equalization pressure, and SA joint aspect-ratio search."""
import numpy as np
import time
import random
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 21
EPS = 1e-9


def _softplus(x, beta=100.0):
    """Numerically stable softplus."""
    xb = beta * x
    return np.where(xb > 0,
                    x + np.log1p(np.exp(-xb)) / beta,
                    np.log1p(np.exp(xb)) / beta)


def _softplus_grad(x, beta=100.0):
    """Gradient of softplus: sigmoid(beta*x)."""
    return 1.0 / (1.0 + np.exp(-beta * x))


def _objective_and_grad(x, w, h, beta=100.0, eq_weight=0.05):
    """
    Softplus-penalized objective with analytical gradients (fully vectorized).
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
        # d/dr_i of eq_weight * mean((rs - r_mean)^2) = eq_weight * 2*(rs_i - r_mean)/N
        grad[2::3] += eq_weight * 2.0 * (rs - r_mean) / N

    # Boundary penalties
    v1 = rs - xs          # x - r < 0  ->  v1 > 0 means violation
    obj += np.sum(_softplus(v1, beta))
    g1 = _softplus_grad(v1, beta)
    grad[0::3] -= g1
    grad[2::3] += g1

    v2 = xs + rs - w      # x + r > w
    obj += np.sum(_softplus(v2, beta))
    g2 = _softplus_grad(v2, beta)
    grad[0::3] += g2
    grad[2::3] += g2

    v3 = rs - ys          # y - r < 0
    obj += np.sum(_softplus(v3, beta))
    g3 = _softplus_grad(v3, beta)
    grad[1::3] -= g3
    grad[2::3] += g3

    v4 = ys + rs - h      # y + r > h
    obj += np.sum(_softplus(v4, beta))
    g4 = _softplus_grad(v4, beta)
    grad[1::3] += g4
    grad[2::3] += g4

    v5 = -rs              # r < 0
    obj += np.sum(_softplus(v5, beta))
    g5 = _softplus_grad(v5, beta)
    grad[2::3] -= g5

    # Pairwise overlap penalties (fully vectorized)
    # Create NxN distance matrix
    dx = xs[:, np.newaxis] - xs[np.newaxis, :]
    dy = ys[:, np.newaxis] - ys[np.newaxis, :]
    d2 = dx * dx + dy * dy
    np.fill_diagonal(d2, 1e-12)  # Avoid division by zero
    d = np.sqrt(d2)
    
    # Sum of radii matrix
    rs_sum = rs[:, np.newaxis] + rs[np.newaxis, :]
    
    # Violation matrix
    v = rs_sum - d
    
    # Softplus of violations
    sp_v = _softplus(v, beta)
    # Gradient of softplus
    g_v = _softplus_grad(v, beta)
    
    # Sum of penalties (exclude diagonal)
    np.fill_diagonal(sp_v, 0)
    np.fill_diagonal(g_v, 0)
    
    obj += np.sum(sp_v)
    
    # Gradients for positions and radii
    # d/dx_i = sum_j g_ij * (x_i - x_j) / d_ij
    # d/dy_i = sum_j g_ij * (y_i - y_j) / d_ij
    # d/dr_i = sum_j g_ij
    
    # Use broadcasting to compute gradients
    # g_v is NxN, dx is NxN, d is NxN
    # grad_x_i = sum_j g_v[i,j] * dx[i,j] / d[i,j]
    grad_x = np.sum(g_v * dx / d, axis=1)
    grad_y = np.sum(g_v * dy / d, axis=1)
    grad_r = np.sum(g_v, axis=1)
    
    grad[0::3] += grad_x
    grad[1::3] += grad_y
    grad[2::3] += grad_r

    # Gradient of -sum(r)
    grad[2::3] += -1.0

    return obj, grad


def _strictly_valid(circles, w, h, tol=1e-8):
    """Full-pair strict validity check. circles: (N,3)."""
    n = circles.shape[0]
    xs = circles[:, 0]
    ys = circles[:, 1]
    rs = circles[:, 2]
    if np.any(rs < 1e-9):
        return False
    if np.any(xs - rs < -tol) or np.any(xs + rs > w + tol):
        return False
    if np.any(ys - rs < -tol) or np.any(ys + rs > h + tol):
        return False
    for i in range(n):
        dx = xs[i] - xs[i + 1:]
        dy = ys[i] - ys[i + 1:]
        d2 = dx * dx + dy * dy
        need = rs[i] + rs[i + 1:] - tol
        if np.any(d2 < need * need):
            return False
    return True


def _delaunay_neighbors(centers):
    """Return list of neighbor-index lists via Delaunay; fallback to all pairs."""
    try:
        tri = Delaunay(centers)
        nbrs = [set() for _ in range(N)]
        for i, j, k in tri.simplices:
            nbrs[i].add(j); nbrs[i].add(k)
            nbrs[j].add(i); nbrs[j].add(k)
            nbrs[k].add(i); nbrs[k].add(j)
        return [list(s) for s in nbrs]
    except Exception:
        return [list(range(N)) for _ in range(N)]


def _fast_valid(circles, w, h, neighbors, tol=1e-9):
    """Fast validity using precomputed neighbor list (approximate, for polish steps)."""
    n = circles.shape[0]
    xs = circles[:, 0]
    ys = circles[:, 1]
    rs = circles[:, 2]
    if np.any(rs < 1e-9):
        return False
    if np.any(xs - rs < -tol) or np.any(xs + rs > w + tol):
        return False
    if np.any(ys - rs < -tol) or np.any(ys + rs > h + tol):
        return False
    for i in range(n):
        for j in neighbors[i]:
            if j <= i:
                continue
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            d2 = dx * dx + dy * dy
            need = rs[i] + rs[j] - tol
            if d2 < need * need:
                return False
    return True


def _hex_init(w, h, seed):
    """Hexagonal lattice init over several row configurations; returns best valid (N,3)."""
    rng = random.Random(seed)
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
        if _strictly_valid(circles, w, h):
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


def _grid_init(w, h, seed):
    rng = random.Random(seed)
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


def _random_init(w, h, seed):
    rng = np.random.RandomState(seed)
    r0 = min(w, h) / 2.0 * 0.25
    circles = np.zeros((N, 3))
    circles[:, 0] = rng.uniform(r0, w - r0, N)
    circles[:, 1] = rng.uniform(r0, h - r0, N)
    circles[:, 2] = r0 * rng.uniform(0.8, 1.2, N)
    return circles


def _optimize_single(circles0, w, h, beta=100.0, max_iter=300, eq_weight=0.05):
    """Run L-BFGS-B from a starting config with given beta; return circles."""
    x0 = circles0.flatten()
    r_max = min(w, h) / 2.0
    bounds = []
    for _ in range(N):
        bounds.append((0.0, w))
        bounds.append((0.0, h))
        bounds.append((1e-6, r_max))
    try:
        res = minimize(
            lambda p: _objective_and_grad(p, w, h, beta, eq_weight),
            x0,
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50},
        )
        return res.x.reshape(-1, 3)
    except Exception:
        return circles0.copy()


def _multi_phase_polish(circles0, w, h, deadline, eq_weight=0.05):
    """
    Multi-phase increasing-beta softplus polish: smooth start -> tight local solve.
    Phases: (beta, max_iter) pairs with increasing beta for tighter convergence.
    """
    circles = circles0.copy()
    phases = [
        (30.0, 100),
        (80.0, 200),
        (150.0, 300),
        (300.0, 400),
        (600.0, 500),
        (1000.0, 600),
    ]
    for beta, iters in phases:
        if time.time() > deadline:
            break
        circles = _optimize_single(circles, w, h, beta=beta, max_iter=iters, eq_weight=eq_weight)
    return circles


def _sa_joint_optimize(circles0, w0, h0, deadline, seed=42):
    """
    Simulated annealing that jointly optimizes circle positions/radii AND aspect ratio.
    Uses a quadratic penalty cost function for speed.
    """
    xs = circles0[:, 0].copy()
    ys = circles0[:, 1].copy()
    rs = circles0[:, 2].copy()
    w = w0
    h = h0
    T = 0.05
    cooling = 0.9997
    rng = np.random.default_rng(seed)
    max_iters = 2000

    def cost(xs, ys, rs, w, h):
        bpen = (np.sum(np.maximum(0, rs - xs) ** 2)
                + np.sum(np.maximum(0, xs - (w - rs)) ** 2)
                + np.sum(np.maximum(0, rs - ys) ** 2)
                + np.sum(np.maximum(0, ys - (h - rs)) ** 2))
        overlap = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                d = np.sqrt(dx * dx + dy * dy + 1e-12)
                viol = max(0, rs[i] + rs[j] - d)
                overlap += viol ** 2
        # Equalization pressure
        r_mean = np.mean(rs)
        eq_pen = 0.05 * np.sum((rs - r_mean) ** 2)
        return bpen + overlap + eq_pen - np.sum(rs)

    current_cost = cost(xs, ys, rs, w, h)
    best_circles = None
    best_cost = current_cost

    for it in range(max_iters):
        if time.time() > deadline:
            break

        new_xs = xs.copy()
        new_ys = ys.copy()
        new_rs = rs.copy()
        new_w = w
        new_h = h

        # Decide what to perturb
        action = rng.random()
        if action < 0.12:
            # Perturb aspect ratio
            new_w = np.clip(w + rng.normal(0, T * 0.3), 0.4, 1.6)
            new_h = 2.0 - new_w
        elif action < 0.40:
            # Perturb one circle's position
            i = rng.integers(0, N)
            new_xs[i] = np.clip(xs[i] + rng.normal(0, T * 0.2), 0, w)
            new_ys[i] = np.clip(ys[i] + rng.normal(0, T * 0.2), 0, h)
        elif action < 0.65:
            # Perturb one circle's radius
            i = rng.integers(0, N)
            new_rs[i] = np.clip(rs[i] + rng.normal(0, T * 0.1), 1e-6, min(w, h) / 2)
        elif action < 0.80:
            # Perturb two circles' positions
            i, j = rng.choice(N, 2, replace=False)
            new_xs[i] = np.clip(xs[i] + rng.normal(0, T * 0.2), 0, w)
            new_ys[i] = np.clip(ys[i] + rng.normal(0, T * 0.2), 0, h)
            new_xs[j] = np.clip(xs[j] + rng.normal(0, T * 0.2), 0, w)
            new_ys[j] = np.clip(ys[j] + rng.normal(0, T * 0.2), 0, h)
        else:
            # Perturb multiple radii slightly
            idx = rng.choice(N, 3, replace=False)
            new_rs[idx] = np.clip(rs[idx] + rng.normal(0, T * 0.08, 3), 1e-6, min(w, h) / 2)

        new_cost = cost(new_xs, new_ys, new_rs, new_w, new_h)
        delta = new_cost - current_cost

        if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
            xs, ys, rs, w, h = new_xs, new_ys, new_rs, new_w, new_h
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_circles = np.column_stack([xs, ys, rs])

        T *= cooling

    if best_circles is not None:
        return best_circles, w, h
    return circles0.copy(), w, h


def _greedy_radius_polish(circles, w, h, deadline):
    """Greedy radius-increase polish using Delaunay fast checks, then strict verify."""
    circles = circles.copy()
    centers = circles[:, :2]
    neighbors = _delaunay_neighbors(centers)
    for _ in range(80):
        if time.time() > deadline:
            break
        improved = False
        for i in range(N):
            old_r = circles[i, 2]
            new_r = old_r * 1.005
            circles[i, 2] = new_r
            if _fast_valid(circles, w, h, neighbors):
                improved = True
            else:
                circles[i, 2] = old_r
        if not improved:
            break
        # Refresh neighbor list occasionally
        if _ % 10 == 0:
            centers = circles[:, :2]
            neighbors = _delaunay_neighbors(centers)
    # Final strict verification; if invalid, roll back radii
    if not _strictly_valid(circles, w, h):
        for i in range(N):
            lo, hi = 1e-6, circles[i, 2]
            for _ in range(25):
                mid = 0.5 * (lo + hi)
                circles[i, 2] = mid
                if _strictly_valid(circles, w, h):
                    lo = mid
                else:
                    hi = mid
            circles[i, 2] = lo
    return circles


def _high_precision_polish(circles, w, h, deadline):
    """Final high-precision polish with very high beta and tight tolerances."""
    if time.time() > deadline - 3:
        return circles
    x0 = circles.flatten()
    r_max = min(w, h) / 2.0
    bounds = []
    for _ in range(N):
        bounds.append((0.0, w))
        bounds.append((0.0, h))
        bounds.append((1e-6, r_max))
    try:
        res = minimize(
            lambda p: _objective_and_grad(p, w, h, 1000.0, 0.01),
            x0,
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,
            options={'maxiter': 600, 'ftol': 1e-15, 'gtol': 1e-11, 'maxls': 60},
        )
        new_circles = res.x.reshape(-1, 3)
        if _strictly_valid(new_circles, w, h):
            return new_circles
    except Exception:
        pass
    return circles


def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s kill

    best_circles = None
    best_value = -np.inf
    best_w = best_h = None

    def try_update(circles, w, h):
        nonlocal best_circles, best_value, best_w, best_h
        if circles is not None and _strictly_valid(circles, w, h):
            val = float(np.sum(circles[:, 2]))
            if val > best_value:
                best_value = val
                best_circles = circles.copy()
                best_w, best_h = w, h
            return val
        return -np.inf

    # Aspect ratios: w + h = 2, so h = 2 - w
    aspect_ratios = [1.0, 1.2, 1.4, 1.6, 1.8, 1.3, 1.5, 1.7,
                     0.8, 0.9, 1.1, 2.0, 0.7, 2.2, 0.65, 2.35, 0.6]

    seed_counter = 0

    for w in aspect_ratios:
        h = 2.0 - w
        if w < 0.4 or h < 0.4:
            continue
        if time.time() > deadline:
            break

        # Multiple diverse initializations per aspect ratio
        inits = [
            _hex_init(w, h, seed_counter),
            _grid_init(w, h, seed_counter + 1000),
            _random_init(w, h, seed_counter + 2000),
        ]
        seed_counter += 50

        for circles0 in inits:
            if time.time() > deadline:
                break

            # Multi-phase polish: smooth -> tight
            circles = _multi_phase_polish(circles0, w, h, deadline, eq_weight=0.05)

            # Greedy radius polish
            circles = _greedy_radius_polish(circles, w, h, deadline)

            val = try_update(circles, w, h)

            # If this is a promising result, try SA joint optimization
            if val > 0 and time.time() < deadline - 15:
                sa_circles, sa_w, sa_h = _sa_joint_optimize(
                    circles, w, h, deadline - 10, seed=seed_counter + 3000
                )
                if sa_circles is not None:
                    sa_circles = _multi_phase_polish(sa_circles, sa_w, sa_h, deadline - 5, eq_weight=0.03)
                    sa_circles = _greedy_radius_polish(sa_circles, sa_w, sa_h, deadline - 2)
                    try_update(sa_circles, sa_w, sa_h)

    # Final high-precision polish on the best found
    if best_circles is not None and time.time() < deadline - 5:
        hp_circles = _high_precision_polish(best_circles, best_w, best_h, deadline)
        try_update(hp_circles, best_w, best_h)

        # One more greedy radius polish after high-precision
        if time.time() < deadline - 2:
            final_circles = _greedy_radius_polish(best_circles, best_w, best_h, deadline)
            try_update(final_circles, best_w, best_h)

    if best_circles is None:
        # Absolute fallback: tiny valid grid
        w, h = 1.4, 0.6
        a, b = 7, 3
        circles = np.zeros((N, 3))
        for i in range(N):
            col = i % a
            row = i // a
            r = min(w / a, h / b) * 0.45
            circles[i, 0] = (col + 0.5) * w / a
            circles[i, 1] = (row + 0.5) * h / b
            circles[i, 2] = r
        best_circles = circles

    return best_circles
