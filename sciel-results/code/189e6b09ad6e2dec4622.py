import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 21
DEADLINE = 130.0

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE

    best_value = 0.0
    best_result = None

    def _rect_dims(w):
        return w, 2.0 - w

    def _vectorized_objective(z):
        x = z[0::3][:N]
        y = z[1::3][:N]
        r = z[2::3][:N]
        w = z[-1]
        h = 2.0 - w

        if w < 0.3 or h < 0.3:
            return 1e6

        obj = -np.sum(r)

        # Pairwise overlap penalty (vectorized)
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d2 = dx * dx + dy * dy
                req = r[i] + r[j]
                if d2 < req * req:
                    d = np.sqrt(d2) if d2 > 0 else 0.0
                    obj += 500.0 * (req - d) ** 2

        # Boundary penalty
        for i in range(N):
            if x[i] < r[i]:
                obj += 500.0 * (r[i] - x[i]) ** 2
            elif x[i] > w - r[i]:
                obj += 500.0 * (x[i] - (w - r[i])) ** 2
            if y[i] < r[i]:
                obj += 500.0 * (r[i] - y[i]) ** 2
            elif y[i] > h - r[i]:
                obj += 500.0 * (y[i] - (h - r[i])) ** 2

        return obj

    def check_validity(z, tol=1e-7):
        x = z[0::3][:N]
        y = z[1::3][:N]
        r = z[2::3][:N]
        w = z[-1]
        h = 2.0 - w

        if w <= 0.05 or h <= 0.05:
            return False, 0.0

        for i in range(N):
            if r[i] <= 0:
                return False, 0.0
            if x[i] < r[i] - tol or x[i] > w - r[i] + tol:
                return False, 0.0
            if y[i] < r[i] - tol or y[i] > h - r[i] + tol:
                return False, 0.0

        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d2 = dx * dx + dy * dy
                r_sum = r[i] + r[j]
                if d2 < r_sum ** 2 - tol:
                    return False, 0.0

        return True, float(np.sum(r))

    def _grid_seed(w0, seed_val=0):
        rng = np.random.default_rng(seed_val)
        h0 = 2.0 - w0
        side = int(np.ceil(np.sqrt(N)))
        z = []
        for i in range(N):
            row, col = divmod(i, side)
            cx = (col + 0.5) / side * w0 + rng.uniform(-0.02, 0.02)
            cy = (row + 0.5) / side * h0 + rng.uniform(-0.02, 0.02)
            r0 = min(w0 / side, h0 / side) * 0.42
            z.extend([cx, cy, r0])
        z.append(w0)
        return np.array(z)

    def _random_seed(rng, w0):
        h0 = 2.0 - w0
        z = []
        for i in range(N):
            cx = rng.uniform(0.05, w0 - 0.05)
            cy = rng.uniform(0.05, h0 - 0.05)
            r0 = rng.uniform(0.04, 0.10)
            z.extend([cx, cy, r0])
        z.append(w0)
        return np.array(z)

    def _force_relax(z, max_iter=200):
        z = z.copy().astype(float)
        n = N
        dt = 0.004

        for _ in range(max_iter):
            x = z[0::3][:n].copy()
            y = z[1::3][:n].copy()
            r = z[2::3][:n]
            w = z[-1]
            h = 2.0 - w

            fx = np.zeros(n)
            fy = np.zeros(n)

            for i in range(n):
                if x[i] - r[i] < 0:
                    fx[i] += (x[i] - r[i]) * 25
                if x[i] + r[i] > w:
                    fx[i] -= (x[i] + r[i] - w) * 25
                if y[i] - r[i] < 0:
                    fy[i] += (y[i] - r[i]) * 25
                if y[i] + r[i] > h:
                    fy[i] -= (y[i] + r[i] - h) * 25

            for i in range(n):
                for j in range(i + 1, n):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d2 = dx * dx + dy * dy
                    min_d = r[i] + r[j]
                    if d2 < min_d * min_d:
                        d = np.sqrt(d2) if d2 > 1e-12 else 1e-12
                        overlap = min_d - d
                        inv_d = 1.0 / d
                        fx[i] += dx * inv_d * overlap * 12
                        fy[i] += dy * inv_d * overlap * 12
                        fx[j] -= dx * inv_d * overlap * 12
                        fy[j] -= dy * inv_d * overlap * 12

            z[0::3][:n] += dt * fx
            z[1::3][:n] += dt * fy

            for i in range(n):
                z[3 * i] = np.clip(z[3 * i], r[i], w - r[i])
                z[3 * i + 1] = np.clip(z[3 * i + 1], r[i], h - r[i])

        return z

    def _polish(z, time_budget=10.0):
        z = z.copy().astype(float)
        n = N
        x = z[0::3][:n].copy()
        y = z[1::3][:n].copy()
        r = z[2::3][:n].copy()
        w = z[-1]
        h = 2.0 - w

        dt = 0.0005
        start = time.time()

        for it in range(800):
            if time.time() - start > time_budget:
                break

            fx = np.zeros(n)
            fy = np.zeros(n)

            for i in range(n):
                if x[i] - r[i] < 0:
                    fx[i] += (x[i] - r[i]) * 60
                if x[i] + r[i] > w:
                    fx[i] -= (x[i] + r[i] - w) * 60
                if y[i] - r[i] < 0:
                    fy[i] += (y[i] - r[i]) * 60
                if y[i] + r[i] > h:
                    fy[i] -= (y[i] + r[i] - h) * 60

            for i in range(n):
                for j in range(i + 1, n):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d2 = dx * dx + dy * dy
                    min_d = r[i] + r[j]
                    if d2 < min_d * min_d:
                        d = np.sqrt(d2) if d2 > 1e-12 else 1e-12
                        overlap = min_d - d
                        inv_d = 1.0 / d
                        fx[i] += dx * inv_d * overlap * 25
                        fy[i] += dy * inv_d * overlap * 25
                        fx[j] -= dx * inv_d * overlap * 25
                        fy[j] -= dy * inv_d * overlap * 25

            x += dt * fx
            y += dt * fy

            # Try to grow radii
            growth = 0.00005
            for i in range(n):
                can_grow = True
                if x[i] - (r[i] + growth) < 0 or x[i] + (r[i] + growth) > w:
                    can_grow = False
                if y[i] - (r[i] + growth) < 0 or y[i] + (r[i] + growth) > h:
                    can_grow = False
                if can_grow:
                    for j in range(n):
                        if i == j:
                            continue
                        dx = x[i] - x[j]
                        dy = y[i] - y[j]
                        d2 = dx * dx + dy * dy
                        if d2 < (r[i] + r[j] + 2 * growth) ** 2:
                            can_grow = False
                            break
                if can_grow:
                    r[i] += growth

            for i in range(n):
                x[i] = np.clip(x[i], r[i], w - r[i])
                y[i] = np.clip(y[i], r[i], h - r[i])

        z_polished = np.zeros_like(z)
        z_polished[0::3][:n] = x
        z_polished[1::3][:n] = y
        z_polished[2::3][:n] = r
        z_polished[-1] = w
        return z_polished

    def _slsqp_optimize(z0, maxiter=300):
        bounds = [(0.0, 2.0)] * (3 * N) + [(0.3, 1.7)]
        try:
            res = minimize(_vectorized_objective, z0, method='SLSQP',
                           bounds=bounds,
                           options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 80})
            return res.x
        except Exception:
            return None

    def _try_update(z):
        nonlocal best_value, best_result
        if z is None:
            return
        valid, val = check_validity(z)
        if valid and val > best_value:
            best_value = val
            best_result = z.copy()
        elif not valid:
            # Try shrinking radii slightly to fix validity
            z_fix = z.copy()
            r_idx = np.arange(2, len(z_fix) - 1, 3)
            for factor in [0.98, 0.95, 0.90]:
                z_fix[r_idx] = z[r_idx] * factor
                valid2, val2 = check_validity(z_fix)
                if valid2 and val2 > best_value:
                    best_value = val2
                    best_result = z_fix.copy()
                    break

    # Generate diverse initial guesses
    rng = np.random.default_rng(123)
    aspect_ratios = [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.75, 0.55, 1.2, 0.9]
    guesses = []
    for idx, w0 in enumerate(aspect_ratios):
        h0 = 2.0 - w0
        if h0 < 0.3:
            continue
        guesses.append(_grid_seed(w0, seed_val=idx))
        guesses.append(_random_seed(rng, w0))

    # Phase 1: Force relax + SLSQP for each guess
    for i, z0 in enumerate(guesses):
        if time.time() > deadline - 8:
            break

        z_relaxed = _force_relax(z0, max_iter=200)
        remaining = deadline - time.time()
        if remaining < 3:
            break

        maxiter = int(min(300, remaining * 30))
        z_opt = _slsqp_optimize(z_relaxed, maxiter=maxiter)
        _try_update(z_opt)

    # Phase 2: Differential evolution for global exploration
    if time.time() < deadline - 15:
        def _de_objective(params):
            z = params.copy()
            x = z[0::3][:N]
            y = z[1::3][:N]
            r = z[2::3][:N]
            w = z[-1]
            h = 2.0 - w
            if w < 0.3 or h < 0.3:
                return 1e6
            obj = -np.sum(r)
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d2 = dx * dx + dy * dy
                    req = r[i] + r[j]
                    if d2 < req * req:
                        d = np.sqrt(d2) if d2 > 0 else 0.0
                        obj += 500.0 * (req - d) ** 2
            for i in range(N):
                if x[i] < r[i]:
                    obj += 500.0 * (r[i] - x[i]) ** 2
                elif x[i] > w - r[i]:
                    obj += 500.0 * (x[i] - (w - r[i])) ** 2
                if y[i] < r[i]:
                    obj += 500.0 * (r[i] - y[i]) ** 2
                elif y[i] > h - r[i]:
                    obj += 500.0 * (y[i] - (h - r[i])) ** 2
            return obj

        bounds_de = [(0.0, 2.0)] * (3 * N) + [(0.5, 1.5)]
        time_budget_de = deadline - time.time() - 10
        if time_budget_de > 5:
            try:
                de_res = differential_evolution(
                    _de_objective, bounds_de,
                    seed=42, maxiter=30, tol=1e-8,
                    popsize=8, polish=True,
                    update_interval=5,
                    init='sobol'
                )
                _try_update(de_res.x)
            except Exception:
                pass

    # Phase 3: Polish the best result
    if best_result is not None and time.time() < deadline - 5:
        remaining = deadline - time.time() - 3
        if remaining > 3:
            polished = _polish(best_result, time_budget=remaining)
            _try_update(polished)

    # Fallback
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    x = best_result[0::3][:N]
    y = best_result[1::3][:N]
    r = best_result[2::3][:N]
    r = r * (1 - 1e-7)

    return np.column_stack((x, y, r))
