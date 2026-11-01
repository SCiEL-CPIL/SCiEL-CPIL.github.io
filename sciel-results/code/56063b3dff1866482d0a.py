"""Circle packing in perimeter-4 rectangle: 21 circles, maximize sum of radii.
Combined: unified penalty-based SLSQP with log-sum-exp smooth penalties +
expansion polish + jiggle refinement + aspect ratio search."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21


def _is_valid(cx, cy, r, W, H, tol=1e-9):
    """Check if packing is valid."""
    if W <= 0.01 or H <= 0.01:
        return False
    if np.any(cx - r < -tol) or np.any(cy - r < -tol):
        return False
    if np.any(cx + r > W + tol) or np.any(cy + r > H + tol):
        return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            if dx * dx + dy * dy < (r[i] + r[j]) ** 2 - tol:
                return False
    return True


def _sum_radii(cx, cy, r, W, H):
    if _is_valid(cx, cy, r, W, H):
        return float(np.sum(r))
    return -1.0


def _expand_radii(x, y, r, w, h, max_iter=300):
    """Iteratively expand each radius to its maximum allowed value."""
    r = r.copy()
    for _ in range(max_iter):
        improved = False
        for i in range(N):
            r_max = min(x[i], w - x[i], y[i], h - y[i])
            for j in range(N):
                if i == j:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                r_allowed = d - r[j]
                if r_allowed < r_max:
                    r_max = r_allowed
            r_max = max(r_max, 0.001)
            if r_max > r[i] + 1e-10:
                r[i] = r_max
                improved = True
        if not improved:
            break
    return r


def _jiggle(x, y, r, w, h, max_iter=500, time_budget=5.0):
    """Random perturbation + local optimization to escape local minima."""
    t0 = time.time()
    best_r = r.copy()
    best_val = _sum_radii(x, y, r, w, h)
    if best_val < 0:
        return r

    rng = np.random.default_rng(123)
    for it in range(max_iter):
        if time.time() - t0 > time_budget:
            break
        eps = 0.002 * (1.0 - it / max_iter)
        x_p = x + rng.uniform(-eps, eps, N)
        y_p = y + rng.uniform(-eps, eps, N)
        x_p = np.clip(x_p, r, w - r)
        y_p = np.clip(y_p, r, h - r)
        r_p = _expand_radii(x_p, y_p, r, w, h)
        val = _sum_radii(x_p, y_p, r_p, w, h)
        if val > best_val:
            best_val = val
            best_r = r_p.copy()
            x = x_p.copy()
            y = y_p.copy()
            r = r_p.copy()
    return best_r


def _logsumexp_penalty(violations, beta=100.0):
    """Smooth log-sum-exp penalty for constraint violations."""
    if len(violations) == 0:
        return 0.0
    v = np.maximum(violations, 0.0)
    if np.max(v) < 1e-12:
        return 0.0
    scaled = beta * v
    max_s = np.max(scaled)
    if max_s > 500:
        return max_s / beta + np.log(np.sum(np.exp(scaled - max_s))) / beta
    return np.log(np.sum(np.exp(scaled))) / beta


def _unified_objective(z, beta=100.0):
    """Unified objective with log-sum-exp smooth penalties.
    z = [x_1,...,x_N, y_1,...,y_N, r_1,...,r_N, w]
    h = 2 - w
    """
    x = z[:N]
    y = z[N:2*N]
    r = z[2*N:3*N]
    w = z[3*N]
    h = 2.0 - w

    if w < 0.3 or h < 0.3:
        return 1e6
    if np.any(r < 0.001):
        return 1e6

    obj = -np.sum(r)

    # Pairwise overlap violations
    iu = np.triu_indices(N, k=1)
    dx = x[iu[0]] - x[iu[1]]
    dy = y[iu[0]] - y[iu[1]]
    d = np.sqrt(dx * dx + dy * dy)
    r_sum = r[iu[0]] + r[iu[1]]
    overlap_viol = r_sum - d
    obj += _logsumexp_penalty(overlap_viol, beta)

    # Boundary violations
    left_viol = r - x
    right_viol = x + r - w
    bottom_viol = r - y
    top_viol = y + r - h
    all_boundary = np.concatenate([left_viol, right_viol, bottom_viol, top_viol])
    obj += _logsumexp_penalty(all_boundary, beta)

    # Keep aspect ratio reasonable
    if w < 0.5:
        obj += 50.0 * (0.5 - w) ** 2
    elif w > 1.5:
        obj += 50.0 * (w - 1.5) ** 2

    return obj


def _force_relax(z, max_iter=300, dt_init=0.003):
    """Simple force-directed relaxation to resolve initial overlaps."""
    z = z.copy().astype(float)
    n = N
    dt = dt_init
    for it in range(max_iter):
        x = z[:n].copy()
        y = z[n:2*n].copy()
        r = z[2*n:3*n].copy()
        w = z[3*n]
        h = 2.0 - w

        fx = np.zeros(n)
        fy = np.zeros(n)
        fr = np.zeros(n)
        fw = 0.0

        # Pairwise repulsion
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                if d < 1e-12:
                    dx, dy, d = 0.01, 0.01, 0.0141
                overlap = r[i] + r[j] - d
                if overlap > 0:
                    fx[i] += (dx / d) * overlap
                    fy[i] += (dy / d) * overlap
                    fx[j] -= (dx / d) * overlap
                    fy[j] -= (dy / d) * overlap
                    fr[i] -= 0.5 * overlap
                    fr[j] -= 0.5 * overlap

        # Boundary forces
        for i in range(n):
            if x[i] < r[i]:
                fx[i] += (r[i] - x[i])
            if x[i] > w - r[i]:
                fx[i] -= (x[i] - (w - r[i]))
            if y[i] < r[i]:
                fy[i] += (r[i] - y[i])
            if y[i] > h - r[i]:
                fy[i] -= (y[i] - (h - r[i]))

        # Update
        z[:n] += dt * fx
        z[n:2*n] += dt * fy
        z[2*n:3*n] = np.maximum(z[2*n:3*n] + dt * fr, 0.001)

        # Clamp positions
        for i in range(n):
            z[i] = np.clip(z[i], z[2*n+i], w - z[2*n+i])
            z[n+i] = np.clip(z[n+i], z[2*n+i], h - z[2*n+i])

        dt *= 0.995
        if dt < 1e-5:
            break

    return z


def _optimize_aspect_ratio(x, y, r, w, h, time_budget=8.0):
    """Try to improve the aspect ratio by adjusting w and re-optimizing."""
    t0 = time.time()
    best = (x.copy(), y.copy(), r.copy(), w, h)
    best_val = _sum_radii(x, y, r, w, h)
    if best_val < 0:
        return best

    for w_new in [w * 0.9, w * 1.1, w - 0.1, w + 0.1, w * 0.8, w * 1.2]:
        if time.time() - t0 > time_budget:
            break
        h_new = 2.0 - w_new
        if h_new < 0.3 or w_new < 0.3:
            continue

        scale_x = w_new / w
        scale_y = h_new / h
        x_new = x * scale_x
        y_new = y * scale_y

        r_new = _expand_radii(x_new, y_new, r, w_new, h_new)
        val = _sum_radii(x_new, y_new, r_new, w_new, h_new)
        if val > best_val:
            best_val = val
            best = (x_new.copy(), y_new.copy(), r_new.copy(), w_new, h_new)

    return best


def _full_refine(x, y, r, w, h, time_budget=20.0):
    """Full refinement: expand radii, try aspect ratios, jiggle."""
    t0 = time.time()
    best_val = _sum_radii(x, y, r, w, h)
    if best_val < 0:
        return x, y, r, w, h

    for _ in range(3):
        if time.time() - t0 > time_budget - 5.0:
            break

        # Expand radii
        r = _expand_radii(x, y, r, w, h)

        # Try aspect ratio
        x, y, r, w, h = _optimize_aspect_ratio(x, y, r, w, h,
                                                time_budget=min(3.0, time_budget - (time.time() - t0)))

        val = _sum_radii(x, y, r, w, h)
        if val > best_val:
            best_val = val

    # Final jiggle
    if time.time() - t0 < time_budget - 3.0:
        r = _jiggle(x, y, r, w, h, max_iter=300,
                    time_budget=max(1.0, time_budget - (time.time() - t0)))
        r = _expand_radii(x, y, r, w, h)

    return x, y, r, w, h


def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    best_value = 0.0
    best_x, best_y, best_r, best_w, best_h = None, None, None, None, None

    def get_initial_guesses():
        guesses = []
        side = int(np.ceil(np.sqrt(N)))

        aspect_ratios = [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6,
                         1.75, 0.55, 2.0, 0.5, 1.2, 0.8, 1.4, 0.7, 1.1, 0.9]
        for w0 in aspect_ratios:
            h0 = 2.0 - w0
            if h0 < 0.3 or w0 < 0.3:
                continue

            # Grid layout
            z = np.zeros(3 * N + 1)
            count = 0
            for row in range(side):
                for col in range(side):
                    if count >= N:
                        break
                    cx = (col + 0.5) / side * w0
                    cy = (row + 0.5) / side * h0
                    r0 = min(w0 / side, h0 / side) * 0.44
                    z[count] = cx
                    z[N + count] = cy
                    z[2 * N + count] = r0
                    count += 1
            z[3 * N] = w0
            guesses.append(z)

            # Staggered grid
            z2 = np.zeros(3 * N + 1)
            count = 0
            n_rows = 4
            n_cols = 6
            cell_w = w0 / n_cols
            cell_h = h0 / n_rows
            for row in range(n_rows):
                for col in range(n_cols):
                    if count >= N:
                        break
                    cx = (col + 0.5) * cell_w
                    cy = (row + 0.5) * cell_h
                    if row % 2 == 1:
                        cx += cell_w * 0.25
                        cx = np.clip(cx, 0.05, w0 - 0.05)
                    r0 = min(cell_w, cell_h) * 0.42
                    z2[count] = cx
                    z2[N + count] = cy
                    z2[2 * N + count] = r0
                    count += 1
            z2[3 * N] = w0
            guesses.append(z2)

        # Random starts
        rng = np.random.default_rng(42)
        for _ in range(15):
            w0 = rng.uniform(0.6, 1.4)
            h0 = 2.0 - w0
            z = np.zeros(3 * N + 1)
            for i in range(N):
                r0 = rng.uniform(0.04, 0.10)
                z[i] = rng.uniform(r0, w0 - r0)
                z[N + i] = rng.uniform(r0, h0 - r0)
                z[2 * N + i] = r0
            z[3 * N] = w0
            guesses.append(z)

        return guesses

    guesses = get_initial_guesses()

    # Sort by initial sum of radii (heuristic for potential)
    def potential(z):
        r = z[2 * N:3 * N]
        w = z[3 * N]
        h = 2.0 - w
        if w < 0.3 or h < 0.3:
            return 0.0
        return float(np.sum(r))

    guesses.sort(key=potential, reverse=True)

    bounds = [(0, 2)] * (3 * N) + [(0.3, 1.7)]

    for i, z0 in enumerate(guesses):
        if time.time() > deadline - 30:
            break

        # Force relax to resolve initial overlaps
        z_relaxed = _force_relax(z0, max_iter=200)

        # SLSQP optimization with log-sum-exp penalty
        remaining = deadline - (time.time() - start_time)
        maxiter = int(min(500, max(100, remaining * 30)))
        try:
            res = minimize(_unified_objective, z_relaxed, method='SLSQP',
                           bounds=bounds,
                           options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 80})
            z_opt = res.x
        except Exception:
            z_opt = z_relaxed

        # Extract and validate
        x = z_opt[:N].copy()
        y = z_opt[N:2*N].copy()
        r = z_opt[2*N:3*N].copy()
        w = z_opt[3*N]
        h = 2.0 - w

        if w < 0.3 or h < 0.3:
            continue

        # Clamp to feasible region
        for k in range(N):
            r[k] = max(r[k], 0.001)
            x[k] = np.clip(x[k], r[k], w - r[k])
            y[k] = np.clip(y[k], r[k], h - r[k])

        # Expand radii
        r = _expand_radii(x, y, r, w, h)

        val = _sum_radii(x, y, r, w, h)

        if val > best_value:
            best_value = val
            best_x, best_y, best_r, best_w, best_h = x.copy(), y.copy(), r.copy(), w, h

    # Phase 2: Polish best result
    if best_x is not None and time.time() < deadline - 15:
        x_f, y_f, r_f, w_f, h_f = _full_refine(best_x, best_y, best_r, best_w, best_h,
                                                time_budget=12.0)
        val = _sum_radii(x_f, y_f, r_f, w_f, h_f)
        if val > best_value:
            best_value = val
            best_x, best_y, best_r, best_w, best_h = x_f.copy(), y_f.copy(), r_f.copy(), w_f, h_f

    # Phase 3: Try re-optimizing from best with different beta
    if best_x is not None and time.time() < deadline - 10:
        z_best = np.concatenate([best_x, best_y, best_r, [best_w]])
        for beta in [50.0, 200.0]:
            if time.time() > deadline - 8:
                break
            try:
                res = minimize(lambda z: _unified_objective(z, beta=beta), z_best,
                               method='SLSQP', bounds=bounds,
                               options={'maxiter': 200, 'ftol': 1e-12, 'maxls': 60})
                z_opt = res.x
                x = z_opt[:N].copy()
                y = z_opt[N:2*N].copy()
                r = z_opt[2*N:3*N].copy()
                w = z_opt[3*N]
                h = 2.0 - w
                if w < 0.3 or h < 0.3:
                    continue
                for k in range(N):
                    r[k] = max(r[k], 0.001)
                    x[k] = np.clip(x[k], r[k], w - r[k])
                    y[k] = np.clip(y[k], r[k], h - r[k])
                r = _expand_radii(x, y, r, w, h)
                val = _sum_radii(x, y, r, w, h)
                if val > best_value:
                    best_value = val
                    best_x, best_y, best_r, best_w, best_h = x.copy(), y.copy(), r.copy(), w, h
            except Exception:
                pass

    # Phase 4: Final quick polish
    if best_x is not None and time.time() < deadline - 4:
        x_f, y_f, r_f, w_f, h_f = _full_refine(best_x, best_y, best_r, best_w, best_h,
                                                time_budget=3.0)
        val = _sum_radii(x_f, y_f, r_f, w_f, h_f)
        if val > best_value:
            best_value = val
            best_x, best_y, best_r, best_w, best_h = x_f.copy(), y_f.copy(), r_f.copy(), w_f, h_f

    # Fallback
    if best_x is None:
        w, h = 1.0, 1.0
        side = int(np.ceil(np.sqrt(N)))
        best_x = np.zeros(N)
        best_y = np.zeros(N)
        best_r = np.zeros(N)
        for i in range(N):
            row, col = divmod(i, side)
            best_x[i] = (col + 0.5) / side
            best_y[i] = (row + 0.5) / side
            best_r[i] = 0.5 / side - 0.01
        best_w, best_h = w, h

    # Final safety shrink
    for _ in range(30):
        r_trial = best_r * (1 - 1e-10)
        if _is_valid(best_x, best_y, r_trial, best_w, best_h):
            best_r = r_trial
        else:
            break

    return np.column_stack((best_x, best_y, best_r))
