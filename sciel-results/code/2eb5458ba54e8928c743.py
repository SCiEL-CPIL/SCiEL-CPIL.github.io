"""Circle packing in perimeter-4 rectangle: 21 circles, maximize sum of radii.
Combines constraint-based SLSQP (inspiration) with pebble-growth initializations
and iterative radius-growth polish (current program)."""
import time
import numpy as np
from scipy.optimize import minimize

N = 21
DEADLINE = 140.0  # Safety margin below 150s limit


def _is_valid(circles, w, h, tol=1e-7):
    n = len(circles)
    xs = circles[:, 0]
    ys = circles[:, 1]
    rs = circles[:, 2]

    if np.any(rs < -tol):
        return False
    if np.any(xs < -rs - tol) or np.any(xs > w + rs + tol):
        return False
    if np.any(ys < -rs - tol) or np.any(ys > h + rs + tol):
        return False

    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            if dx * dx + dy * dy < (rs[i] + rs[j]) ** 2 - tol:
                return False
    return True


def _pebble_grow(x, y, W, H, r0=0.015, max_iter=3000, dt=0.0008):
    """Pebble growth algorithm: start with small circles, grow them by
    the minimum distance to nearest neighbor or boundary."""
    xs = x.copy()
    ys = y.copy()
    rs = np.full(N, r0)

    for it in range(max_iter):
        min_dists = np.full(N, np.inf)

        for i in range(N):
            d_left = xs[i] - rs[i]
            d_right = W - xs[i] - rs[i]
            d_bottom = ys[i] - rs[i]
            d_top = H - ys[i] - rs[i]
            min_dists[i] = min(d_left, d_right, d_bottom, d_top)

            for j in range(N):
                if i == j:
                    continue
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                d = np.sqrt(dx * dx + dy * dy)
                min_dists[i] = min(min_dists[i], d - rs[i] - rs[j])

        min_min = np.min(min_dists)
        if min_min <= 1e-8:
            break

        growth = min(min_min, dt)
        rs += growth

        if growth < 1e-7:
            break

    return xs, ys, rs


def _force_relax_positions(x, y, rs, W, H, max_iter=200, dt=0.003):
    """Force-directed relaxation to resolve overlaps and push circles
    away from boundaries."""
    xs = x.copy()
    ys = y.copy()
    r = rs.copy()

    for _ in range(max_iter):
        if time.time() > DEADLINE:
            break
        fx = np.zeros(N)
        fy = np.zeros(N)

        for i in range(N):
            if xs[i] - r[i] < 0:
                fx[i] += (xs[i] - r[i]) * 30
            if xs[i] + r[i] > W:
                fx[i] -= (xs[i] + r[i] - W) * 30
            if ys[i] - r[i] < 0:
                fy[i] += (ys[i] - r[i]) * 30
            if ys[i] + r[i] > H:
                fy[i] -= (ys[i] + r[i] - H) * 30

        for i in range(N):
            for j in range(i + 1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                d = np.sqrt(dx * dx + dy * dy)
                min_d = r[i] + r[j]
                if d < min_d and d > 1e-12:
                    overlap = min_d - d
                    fx[i] += (dx / d) * overlap * 15
                    fy[i] += (dy / d) * overlap * 15
                    fx[j] -= (dx / d) * overlap * 15
                    fy[j] -= (dy / d) * overlap * 15

        xs += dt * fx
        ys += dt * fy

        for i in range(N):
            xs[i] = np.clip(xs[i], r[i], W - r[i])
            ys[i] = np.clip(ys[i], r[i], H - r[i])

    return xs, ys


def _polish(circles, W, H, time_budget=8.0):
    """Polish a valid packing by iteratively growing each radius to its local
    slack and re-relaxing positions. Extracts the last bit of room."""
    t0 = time.time()
    xs = circles[:, 0].copy()
    ys = circles[:, 1].copy()
    rs = circles[:, 2].copy()

    max_outer = 40
    for outer in range(max_outer):
        if time.time() - t0 > time_budget:
            break

        grew = False
        for i in range(N):
            # Boundary slack
            slack = min(xs[i] - rs[i], W - xs[i] - rs[i],
                        ys[i] - rs[i], H - ys[i] - rs[i])

            # Pairwise slack
            for j in range(N):
                if i == j:
                    continue
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                d = np.sqrt(dx * dx + dy * dy)
                slack = min(slack, d - rs[i] - rs[j])

            if slack > 1e-7:
                growth = min(0.003, slack * 0.4)
                rs[i] += growth
                grew = True

        # Light position relaxation after growing
        for _ in range(25):
            if time.time() - t0 > time_budget:
                break
            fx = np.zeros(N)
            fy = np.zeros(N)

            for i in range(N):
                if xs[i] - rs[i] < 0:
                    fx[i] += (xs[i] - rs[i]) * 30
                if xs[i] + rs[i] > W:
                    fx[i] -= (xs[i] + rs[i] - W) * 30
                if ys[i] - rs[i] < 0:
                    fy[i] += (ys[i] - rs[i]) * 30
                if ys[i] + rs[i] > H:
                    fy[i] -= (ys[i] + rs[i] - H) * 30

            for i in range(N):
                for j in range(i + 1, N):
                    dx = xs[i] - xs[j]
                    dy = ys[i] - ys[j]
                    d = np.sqrt(dx * dx + dy * dy)
                    min_d = rs[i] + rs[j]
                    if d < min_d and d > 1e-12:
                        overlap = min_d - d
                        fx[i] += (dx / d) * overlap * 15
                        fy[i] += (dy / d) * overlap * 15
                        fx[j] -= (dx / d) * overlap * 15
                        fy[j] -= (dy / d) * overlap * 15

            dt = 0.001
            xs += dt * fx
            ys += dt * fy

            for i in range(N):
                xs[i] = np.clip(xs[i], rs[i], W - rs[i])
                ys[i] = np.clip(ys[i], rs[i], H - rs[i])

        if not grew:
            break

    circles_out = np.column_stack([xs, ys, rs])
    if _is_valid(circles_out, W, H):
        return circles_out
    return circles


def construct():
    t0 = time.time()
    best_circles = None
    best_val = -np.inf
    best_W = 1.0
    best_H = 1.0

    # --- SLSQP constraint-based optimization ---
    # Parameter vector: [x1, y1, r1, ..., xN, yN, rN, w]
    # h = 2.0 - w

    def objective(z):
        r = z[2::3][:N]
        return -np.sum(r)

    def cons_inside(z):
        x = z[0::3][:N]
        y = z[1::3][:N]
        r = z[2::3][:N]
        w = z[-1]
        h = 2.0 - w
        return np.concatenate([
            x - r,
            w - x - r,
            y - r,
            h - y - r
        ])

    def cons_non_overlap(z):
        x = z[0::3][:N]
        y = z[1::3][:N]
        r = z[2::3][:N]
        cons = []
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                cons.append(dist - r[i] - r[j])
        return np.array(cons)

    def cons_positive_r(z):
        r = z[2::3][:N]
        return r

    def cons_aspect(z):
        w = z[-1]
        h = 2.0 - w
        return np.array([w - 0.35, 1.65 - w])

    constraints = [
        {'type': 'ineq', 'fun': cons_inside},
        {'type': 'ineq', 'fun': cons_non_overlap},
        {'type': 'ineq', 'fun': cons_positive_r},
        {'type': 'ineq', 'fun': cons_aspect}
    ]

    def check_and_extract(z):
        x = z[0::3][:N]
        y = z[1::3][:N]
        r = z[2::3][:N]
        w = z[-1]
        h = 2.0 - w

        if w <= 0.1 or h <= 0.1:
            return None, 0.0, w, h

        circles = np.column_stack([x, y, r])

        # Clip to valid bounds
        for i in range(N):
            max_r = min(circles[i, 0], circles[i, 1],
                        w - circles[i, 0], h - circles[i, 1])
            circles[i, 2] = min(circles[i, 2], max_r)
            circles[i, 2] = max(circles[i, 2], 0.0)

        if _is_valid(circles, w, h):
            return circles, float(np.sum(circles[:, 2])), w, h
        return None, 0.0, w, h

    def _grid_init(W, H, n_cols, n_rows):
        xs = (np.arange(n_cols) + 0.5) * (W / n_cols)
        ys = (np.arange(n_rows) + 0.5) * (H / n_rows)
        X, Y = np.meshgrid(xs, ys, indexing='xy')
        X = X.flatten()[:N]
        Y = Y.flatten()[:N]
        r0 = min(W / n_cols, H / n_rows) / 2.0 * 0.85
        R = np.full(N, r0)
        z = np.zeros(3 * N + 1)
        z[0::3][:N] = X
        z[1::3][:N] = Y
        z[2::3][:N] = R
        z[-1] = W
        return z

    def _random_init(W, H, seed=None):
        rng = np.random.default_rng(seed)
        xs = rng.uniform(0.08 * W, 0.92 * W, N)
        ys = rng.uniform(0.08 * H, 0.92 * H, N)
        rs = np.full(N, 0.06)
        z = np.zeros(3 * N + 1)
        z[0::3][:N] = xs
        z[1::3][:N] = ys
        z[2::3][:N] = rs
        z[-1] = W
        return z

    def _pebble_init(W, H, seed=None):
        rng = np.random.default_rng(seed)
        xs = rng.uniform(0.12 * W, 0.88 * W, N)
        ys = rng.uniform(0.12 * H, 0.88 * H, N)
        xs, ys, rs = _pebble_grow(xs, ys, W, H, r0=0.01, max_iter=1500, dt=0.0006)
        z = np.zeros(3 * N + 1)
        z[0::3][:N] = xs
        z[1::3][:N] = ys
        z[2::3][:N] = rs
        z[-1] = W
        return z

    # Generate diverse initializations
    W_starts = [1.0, 0.85, 1.15, 0.75, 1.25, 0.65, 1.35, 0.9, 1.1, 0.7, 1.3, 0.8]
    all_inits = []

    for W0 in W_starts:
        H0 = 2.0 - W0
        if H0 < 0.35:
            continue

        # Grid initializations
        grid_configs = [(5, 5), (6, 4), (4, 6), (7, 3), (3, 7)]
        for nc, nr in grid_configs:
            if nc * nr >= N:
                all_inits.append(_grid_init(W0, H0, nc, nr))

        # Random initializations
        for seed in range(3):
            all_inits.append(_random_init(W0, H0, seed=seed))

        # Pebble growth initializations
        for seed in range(2):
            all_inits.append(_pebble_init(W0, H0, seed=seed))

    # Shuffle for diversity
    rng_master = np.random.default_rng(99)
    perm = rng_master.permutation(len(all_inits))
    all_inits = [all_inits[i] for i in perm]

    bounds = [(0, 2.0)] * N + [(0, 2.0)] * N + [(0.001, 1.0)] * N + [(0.35, 1.65)]

    for idx, x0 in enumerate(all_inits):
        if time.time() - t0 > DEADLINE - 15:
            break

        try:
            # Stage 1: SLSQP optimization
            res = minimize(
                objective, x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 200, 'ftol': 1e-10, 'disp': 0}
            )

            x_current = res.x
            circles, val, w_cur, h_cur = check_and_extract(x_current)

            if circles is not None and val > best_val:
                best_val = val
                best_circles = circles.copy()
                best_W = w_cur
                best_H = h_cur

            # Stage 2: Polish with radius growth
            if best_circles is not None and time.time() - t0 < DEADLINE - 10:
                polished = _polish(best_circles, best_W, best_H, time_budget=6.0)
                if _is_valid(polished, best_W, best_H):
                    val_pol = float(np.sum(polished[:, 2]))
                    if val_pol > best_val:
                        best_val = val_pol
                        best_circles = polished.copy()

            # Stage 3: Re-optimize from polished positions if time allows
            if best_circles is not None and time.time() - t0 < DEADLINE - 20:
                z_reopt = np.zeros(3 * N + 1)
                z_reopt[0::3][:N] = best_circles[:, 0]
                z_reopt[1::3][:N] = best_circles[:, 1]
                z_reopt[2::3][:N] = best_circles[:, 2]
                z_reopt[-1] = best_W

                res2 = minimize(
                    objective, z_reopt,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 150, 'ftol': 1e-12, 'disp': 0}
                )
                circles2, val2, w2, h2 = check_and_extract(res2.x)
                if circles2 is not None and val2 > best_val:
                    best_val = val2
                    best_circles = circles2.copy()
                    best_W = w2
                    best_H = h2

                # Final polish
                if time.time() - t0 < DEADLINE - 5:
                    polished2 = _polish(best_circles, best_W, best_H, time_budget=5.0)
                    if _is_valid(polished2, best_W, best_H):
                        val_pol2 = float(np.sum(polished2[:, 2]))
                        if val_pol2 > best_val:
                            best_val = val_pol2
                            best_circles = polished2.copy()

        except Exception:
            continue

    # Fallback: simple grid
    if best_circles is None:
        W = H = 1.0
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        best_circles = np.array(cells[:N])

    return best_circles
