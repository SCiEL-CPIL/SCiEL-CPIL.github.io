"""Circle packing in perimeter-4 rectangle: maximize sum of radii for 21 circles."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    best_value = 0.0
    best_result = None

    def check_validity(x, y, r, w, h, tol=1e-8):
        if w <= 0.05 or h <= 0.05:
            return False, 0.0
        if np.any(r <= 0):
            return False, 0.0
        if np.any(x - r < -tol) or np.any(x + r > w + tol):
            return False, 0.0
        if np.any(y - r < -tol) or np.any(y + r > h + tol):
            return False, 0.0
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                if dx * dx + dy * dy < (r[i] + r[j]) ** 2 - tol:
                    return False, 0.0
        return True, float(np.sum(r))

    def _force_relax(x, y, r, w, h, max_iter=800):
        """Force-directed relaxation to resolve overlaps."""
        x = x.copy().astype(float)
        y = y.copy().astype(float)
        dt = 0.005
        for it in range(max_iter):
            if time.time() > deadline - 10:
                break
            fx = np.zeros(N)
            fy = np.zeros(N)
            # Wall repulsion
            fx -= np.maximum(r - x, 0) * 100
            fx += np.maximum(x + r - w, 0) * 100
            fy -= np.maximum(r - y, 0) * 100
            fy += np.maximum(y + r - h, 0) * 100
            # Pairwise repulsion
            max_overlap = 0.0
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx * dx + dy * dy)
                    overlap = r[i] + r[j] - d
                    if overlap > 0:
                        max_overlap = max(max_overlap, overlap)
                        if d > 1e-12:
                            ux = dx / d
                            uy = dy / d
                        else:
                            ux, uy = 1.0, 0.0
                        f = overlap * 15.0
                        fx[i] += ux * f
                        fy[i] += uy * f
                        fx[j] -= ux * f
                        fy[j] -= uy * f
            if max_overlap < 1e-8:
                break
            force_mag = float(np.max(np.sqrt(fx * fx + fy * fy)))
            if force_mag > 1.0:
                dt = max(0.0001, dt * 0.5)
            elif force_mag < 0.1:
                dt = min(0.01, dt * 1.1)
            x += dt * fx
            y += dt * fy
            x = np.clip(x, r, w - r)
            y = np.clip(y, r, h - r)
        return x, y

    def _expand_radii(x, y, r, w, h, time_budget):
        """Iteratively expand each circle to max valid radius."""
        improved = True
        while improved and time.time() < time_budget:
            improved = False
            for i in range(N):
                # Max radius from walls
                r_wall = min(x[i], w - x[i], y[i], h - y[i])
                r_max = r_wall
                for j in range(N):
                    if i == j:
                        continue
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx * dx + dy * dy)
                    r_allowed = d - r[j]
                    if r_allowed < r_max:
                        r_max = r_allowed
                r_max = max(0.0, r_max)
                if r_max > r[i] + 1e-7:
                    # Binary search
                    lo, hi = r[i], r_max
                    for _ in range(15):
                        mid = (lo + hi) / 2
                        valid = True
                        for j in range(N):
                            if i == j:
                                continue
                            dx = x[i] - x[j]
                            dy = y[i] - y[j]
                            if dx * dx + dy * dy < (mid + r[j]) ** 2 - 1e-10:
                                valid = False
                                break
                        if valid and mid <= r_wall + 1e-10:
                            lo = mid
                        else:
                            hi = mid
                    if lo > r[i] + 1e-8:
                        r[i] = lo
                        improved = True
        return r

    def _slsqp_optimize(x0, y0, r0, w0, time_budget):
        """SLSQP optimization with penalty method."""
        def objective(z):
            x = z[0::3][:N]
            y = z[1::3][:N]
            r = z[2::3][:N]
            w = z[-1]
            h = 2.0 - w
            if w < 0.2 or h < 0.2 or np.any(r < 0):
                return 1e6
            obj = -np.sum(r)
            # Overlap penalty
            pen = 0.0
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx * dx + dy * dy)
                    ov = max(0, r[i] + r[j] - d)
                    pen += ov * ov
            obj += 1000 * pen
            # Boundary penalty
            bp = 0.0
            for i in range(N):
                bp += max(0, r[i] - x[i]) ** 2
                bp += max(0, x[i] + r[i] - w) ** 2
                bp += max(0, r[i] - y[i]) ** 2
                bp += max(0, y[i] + r[i] - h) ** 2
            obj += 1000 * bp
            return obj

        z0 = np.zeros(3 * N + 1)
        z0[0::3][:N] = x0
        z0[1::3][:N] = y0
        z0[2::3][:N] = r0
        z0[-1] = w0

        bounds = [(0, 2)] * (3 * N) + [(0.3, 1.7)]
        maxiter = int(min(500, max(100, (time_budget - (time.time() - start_time)) * 40)))
        try:
            res = minimize(objective, z0, method='SLSQP', bounds=bounds,
                           options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 60})
            z_opt = res.x
            x = z_opt[0::3][:N]
            y = z_opt[1::3][:N]
            r = z_opt[2::3][:N]
            w = z_opt[-1]
            h = 2.0 - w
            valid, val = check_validity(x, y, r, w, h)
            if valid:
                return x, y, r, w, h, val
        except Exception:
            pass
        return None

    # Generate initial guesses with diverse aspect ratios
    def get_initial_guesses():
        guesses = []
        side = int(np.ceil(np.sqrt(N)))
        aspect_ratios = [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.75, 0.55, 1.2, 0.8]
        for w0 in aspect_ratios:
            h0 = 2.0 - w0
            if h0 < 0.3:
                continue
            x = np.zeros(N)
            y = np.zeros(N)
            r = np.zeros(N)
            for i in range(N):
                row, col = divmod(i, side)
                x[i] = (col + 0.5) / side * w0
                y[i] = (row + 0.5) / side * h0
                r[i] = min(w0 / side, h0 / side) * 0.4
            guesses.append((x, y, r, w0, h0))
        # Random starts
        rng = np.random.default_rng(42)
        for _ in range(8):
            w0 = rng.uniform(0.6, 1.4)
            h0 = 2.0 - w0
            x = np.zeros(N)
            y = np.zeros(N)
            r = np.zeros(N)
            for i in range(N):
                r[i] = rng.uniform(0.04, 0.1)
                x[i] = rng.uniform(r[i], w0 - r[i])
                y[i] = rng.uniform(r[i], h0 - r[i])
            guesses.append((x, y, r, w0, h0))
        return guesses

    guesses = get_initial_guesses()

    for i, (x0, y0, r0, w0, h0) in enumerate(guesses):
        if time.time() > deadline - 30:
            break

        # Force relax to resolve overlaps
        x_rel, y_rel = _force_relax(x0, y0, r0, w0, h0, max_iter=400)

        # Expand radii
        r_exp = _expand_radii(x_rel, y_rel, r0.copy(), w0, h0, deadline - 20)

        # SLSQP optimize
        remaining = deadline - (time.time() - start_time) - 15
        if remaining < 5:
            break
        result = _slsqp_optimize(x_rel, y_rel, r_exp, w0, deadline - 10)
        if result is not None:
            x, y, r, w, h, val = result
            if val > best_value:
                best_value = val
                best_result = (x.copy(), y.copy(), r.copy(), w, h)

    # Polish best result
    if best_result is not None and time.time() < deadline - 20:
        x, y, r, w, h = best_result

        # Force-directed polish
        x, y = _force_relax(x, y, r, w, h, max_iter=2000)

        # Expand radii
        r = _expand_radii(x, y, r, w, h, deadline - 10)

        valid, val = check_validity(x, y, r, w, h)
        if valid and val > best_value:
            best_value = val
            best_result = (x.copy(), y.copy(), r.copy(), w, h)

        # Another SLSQP pass
        if time.time() < deadline - 15:
            result = _slsqp_optimize(x, y, r, w, deadline - 8)
            if result is not None:
                x2, y2, r2, w2, h2, val2 = result
                if val2 > best_value:
                    best_value = val2
                    best_result = (x2.copy(), y2.copy(), r2.copy(), w2, h2)

        # Final expansion
        if best_result is not None and time.time() < deadline - 5:
            x, y, r, w, h = best_result
            r = _expand_radii(x, y, r, w, h, deadline - 2)
            valid, val = check_validity(x, y, r, w, h)
            if valid and val > best_value:
                best_value = val
                best_result = (x.copy(), y.copy(), r.copy(), w, h)

    if best_result is None:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    x, y, r, w, h = best_result
    return np.column_stack((x, y, r))
