"""Circle packing in perimeter-4 rectangle: differential evolution + L-BFGS-B + local polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21


def _is_valid(cx, cy, r, W, H, tol=1e-8):
    """Check if packing is valid (vectorized)."""
    if W <= 0.05 or H <= 0.05:
        return False
    if np.any(cx - r < -tol) or np.any(cy - r < -tol):
        return False
    if np.any(cx + r > W + tol) or np.any(cy + r > H + tol):
        return False

    n = len(cx)
    if n < 2:
        return True

    dx = cx[:, None] - cx[None, :]
    dy = cy[:, None] - cy[None, :]
    dist_sq = dx * dx + dy * dy
    r_sum = r[:, None] + r[None, :]

    diff = dist_sq - r_sum * r_sum
    diff[np.diag_indices(n)] = np.inf
    min_diff = np.min(diff)

    if min_diff < -tol:
        return False

    return True


def _compute_value(cx, cy, r, W, H):
    """Return sum of radii if valid, else -inf."""
    if _is_valid(cx, cy, r, W, H):
        return float(np.sum(r))
    return -np.inf


def _penalty_objective(z, penalty=500.0):
    """Penalty-based objective for unconstrained optimizers."""
    n = N
    x = z[0::3][:n]
    y = z[1::3][:n]
    r = z[2::3][:n]
    w = z[-1]
    h = 2.0 - w

    if w < 0.2 or h < 0.2:
        return 1e8
    if np.any(r < 0):
        return 1e8

    obj = -np.sum(r)

    # Pairwise overlap penalty
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d_sq = dx * dx + dy * dy
    r_sum = r[:, None] + r[None, :]
    iu = np.triu_indices(n, k=1)
    d_pairs = np.sqrt(np.maximum(d_sq[iu], 0.0))
    overlap = np.maximum(r_sum[iu] - d_pairs, 0.0)
    obj += penalty * np.sum(overlap ** 2)

    # Boundary penalty
    left_v = np.maximum(r - x, 0.0)
    right_v = np.maximum(x + r - w, 0.0)
    bot_v = np.maximum(r - y, 0.0)
    top_v = np.maximum(y + r - h, 0.0)
    obj += penalty * (np.sum(left_v**2) + np.sum(right_v**2) + np.sum(bot_v**2) + np.sum(top_v**2))

    # Aspect ratio penalty
    if w < 0.5:
        obj += penalty * (0.5 - w) ** 2
    elif w > 1.5:
        obj += penalty * (w - 1.5) ** 2

    return obj


def _local_polish(x, y, r, w, h, time_budget):
    """Local search: try to expand each circle, then move to create space."""
    n = N
    x = x.copy()
    y = y.copy()
    r = r.copy()

    for _ in range(200):
        if time.time() > time_budget:
            break
        improved = False
        for i in range(n):
            # Try to expand radius i
            r_wall = min(x[i], w - x[i], y[i], h - y[i])
            r_max = r_wall
            for j in range(n):
                if i == j:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                r_allowed = d - r[j]
                if r_allowed < r_max:
                    r_max = r_allowed
            r_max = max(r[i], r_max)

            # Binary search for max valid radius
            lo, hi = r[i], r_max
            for _ in range(15):
                mid = (lo + hi) / 2.0
                r_test = r.copy()
                r_test[i] = mid
                if _is_valid(x, y, r_test, w, h):
                    lo = mid
                else:
                    hi = mid
            if lo > r[i] + 1e-9:
                r[i] = lo
                improved = True

        # Try to move circles to create more space
        for i in range(n):
            # Find most constraining neighbor or wall
            best_move = None
            best_gain = 0
            for dx_dir in [-1, 0, 1]:
                for dy_dir in [-1, 0, 1]:
                    if dx_dir == 0 and dy_dir == 0:
                        continue
                    for step in [0.03, 0.015, 0.008, 0.004]:
                        nx = x[i] + dx_dir * step
                        ny = y[i] + dy_dir * step
                        if nx < r[i] or nx > w - r[i] or ny < r[i] or ny > h - r[i]:
                            continue
                        x_test = x.copy()
                        y_test = y.copy()
                        x_test[i] = nx
                        y_test[i] = ny
                        # Check if this move allows larger radius
                        r_wall = min(nx, w - nx, ny, h - ny)
                        r_max = r_wall
                        for j in range(n):
                            if i == j:
                                continue
                            ddx = nx - x[j]
                            ddy = ny - y[j]
                            d = np.sqrt(ddx * ddx + ddy * ddy)
                            r_allowed = d - r[j]
                            if r_allowed < r_max:
                                r_max = r_allowed
                        r_max = max(0, r_max)
                        gain = r_max - r[i]
                        if gain > best_gain:
                            best_gain = gain
                            best_move = (nx, ny)
                            break
                    if best_move:
                        break
                if best_move:
                    break
            if best_move and best_gain > 1e-6:
                x[i] = best_move[0]
                y[i] = best_move[1]
                improved = True

        if not improved:
            break

    # Final uniform expansion
    for _ in range(200):
        if time.time() > time_budget:
            break
        r_trial = r * 1.0005
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    if _is_valid(x, y, r, w, h):
        return x, y, r
    return x, y, r


def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    best_value = 0.0
    best_x, best_y, best_r, best_w, best_h = None, None, None, None, None

    def update_best(x, y, r, w, h):
        nonlocal best_value, best_x, best_y, best_r, best_w, best_h
        val = _compute_value(x, y, r, w, h)
        if val > best_value:
            best_value = val
            best_x, best_y, best_r, best_w, best_h = x.copy(), y.copy(), r.copy(), w, h

    # Initial grid-based packings with different aspect ratios
    def make_grid_packing(w0, n_rows=None, n_cols=None):
        h0 = 2.0 - w0
        if n_rows is None:
            n_rows = 5
            n_cols = 5
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        for i in range(N):
            row = i // n_cols
            col = i % n_cols
            cx = (col + 0.5) / n_cols * w0
            cy = (row + 0.5) / n_rows * h0
            r0 = min(w0 / n_cols, h0 / n_rows) * 0.48
            x[i] = cx
            y[i] = cy
            r[i] = r0
        return x, y, r, w0, h0

    # Try several aspect ratios
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.65, 1.5, 0.6, 1.6, 0.55]
    for w0 in aspect_ratios:
        h0 = 2.0 - w0
        if h0 < 0.3:
            continue
        x, y, r, w, h = make_grid_packing(w0)
        # Quick local polish
        x, y, r = _local_polish(x, y, r, w, h, deadline - 5)
        update_best(x, y, r, w, h)

    # Differential Evolution
    if time.time() < deadline - 60:
        rng = np.random.default_rng(12345)
        pop_size = 40
        dim = 3 * N + 1

        # Bounds
        bounds = [(0.0, 2.0)] * (3 * N) + [(0.3, 1.7)]

        # Initialize population
        pop = np.zeros((pop_size, dim))
        for i in range(pop_size):
            w0 = rng.uniform(0.5, 1.5)
            h0 = 2.0 - w0
            for j in range(N):
                r0 = rng.uniform(0.03, 0.1)
                cx = rng.uniform(r0, w0 - r0)
                cy = rng.uniform(r0, h0 - r0)
                pop[i, 3*j] = cx
                pop[i, 3*j+1] = cy
                pop[i, 3*j+2] = r0
            pop[i, -1] = w0

        # Evaluate initial population
        pop_values = np.full(pop_size, -np.inf)
        for i in range(pop_size):
            z = pop[i]
            x = z[0::3][:N]
            y = z[1::3][:N]
            r = z[2::3][:N]
            w = z[-1]
            h = 2.0 - w
            pop_values[i] = _compute_value(x, y, r, w, h)

        # DE loop
        F = 0.8
        CR = 0.9
        max_gen = 30
        for gen in range(max_gen):
            if time.time() > deadline - 50:
                break

            for i in range(pop_size):
                # Select 3 distinct random individuals
                a, b, c = rng.choice(pop_size, size=3, replace=False)
                while a == b or a == i or b == i:
                    a, b, c = rng.choice(pop_size, size=3, replace=False)

                # Mutation
                v = pop[a] + F * (pop[b] - pop[c])

                # Crossover
                trial = pop[i].copy()
                for d in range(dim):
                    if rng.random() < CR or d == rng.randint(dim):
                        trial[d] = v[d]

                # Clamp to bounds
                for d in range(dim):
                    lo, hi = bounds[d]
                    trial[d] = np.clip(trial[d], lo, hi)

                # Evaluate
                x = trial[0::3][:N]
                y = trial[1::3][:N]
                r = trial[2::3][:N]
                w = trial[-1]
                h = 2.0 - w
                val = _compute_value(x, y, r, w, h)

                if val > pop_values[i]:
                    pop[i] = trial
                    pop_values[i] = val

            # Track best
            best_idx = np.argmax(pop_values)
            if pop_values[best_idx] > best_value:
                z = pop[best_idx]
                x = z[0::3][:N]
                y = z[1::3][:N]
                r = z[2::3][:N]
                w = z[-1]
                h = 2.0 - w
                update_best(x, y, r, w, h)

    # L-BFGS-B refinement from best
    if best_x is not None and time.time() < deadline - 30:
        z0 = np.zeros(3 * N + 1)
        z0[0::3][:N] = best_x
        z0[1::3][:N] = best_y
        z0[2::3][:N] = best_r
        z0[-1] = best_w

        # Try L-BFGS-B with penalty objective
        try:
            res = minimize(_penalty_objective, z0, method='L-BFGS-B',
                           bounds=[(0, 2)] * (3 * N) + [(0.3, 1.7)],
                           options={'maxiter': 500, 'ftol': 1e-14, 'maxls': 100})
            z_opt = res.x
            x = z_opt[0::3][:N]
            y = z_opt[1::3][:N]
            r = z_opt[2::3][:N]
            w = z_opt[-1]
            h = 2.0 - w
            # Polish
            x, y, r = _local_polish(x, y, r, w, h, deadline - 10)
            update_best(x, y, r, w, h)
        except Exception:
            pass

        # SLSQP refinement
        if time.time() < deadline - 15:
            try:
                res2 = minimize(_penalty_objective, z0, method='SLSQP',
                                bounds=[(0, 2)] * (3 * N) + [(0.3, 1.7)],
                                options={'maxiter': 400, 'ftol': 1e-12, 'maxls': 80})
                z_opt2 = res2.x
                x = z_opt2[0::3][:N]
                y = z_opt2[1::3][:N]
                r = z_opt2[2::3][:N]
                w = z_opt2[-1]
                h = 2.0 - w
                x, y, r = _local_polish(x, y, r, w, h, deadline - 5)
                update_best(x, y, r, w, h)
            except Exception:
                pass

    # Multi-start local polish from perturbed best
    if best_x is not None and time.time() < deadline - 10:
        rng = np.random.default_rng(777)
        for _ in range(5):
            if time.time() > deadline - 5:
                break
            x_p = best_x.copy()
            y_p = best_y.copy()
            r_p = best_r.copy()
            idxs = rng.choice(N, size=3, replace=False)
            for idx in idxs:
                x_p[idx] += rng.normal(0, 0.03)
                y_p[idx] += rng.normal(0, 0.03)
            x_p = np.clip(x_p, r_p, best_w - r_p)
            y_p = np.clip(y_p, r_p, best_h - r_p)
            x_p, y_p, r_p = _local_polish(x_p, y_p, r_p, best_w, best_h, deadline - 5)
            update_best(x_p, y_p, r_p, best_w, best_h)

    # Final polish on best
    if best_x is not None and time.time() < deadline - 3:
        best_x, best_y, best_r = _local_polish(best_x, best_y, best_r, best_w, best_h, deadline - 1)
        update_best(best_x, best_y, best_r, best_w, best_h)

    if best_x is None:
        # Fallback
        side = 5
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    # Final safety shrink
    r_final = best_r
    for _ in range(20):
        r_trial = r_final * (1 - 1e-9)
        if _is_valid(best_x, best_y, r_trial, best_w, best_h):
            r_final = r_trial
        else:
            break

    return np.column_stack((best_x, best_y, r_final))
