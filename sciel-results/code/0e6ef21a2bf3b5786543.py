"""Circle packing in perimeter-4 rectangle: Hard-constraint SLSQP with radial coordinates."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21
_IU = np.triu_indices(N, k=1)


def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    best_value = 0.0
    best_result = None

    def check_validity(x, y, r, w, h, tol=1e-9):
        if w <= 0.01 or h <= 0.01:
            return False, 0.0
        if np.any(r < 0):
            return False, 0.0
        if np.any(x < r - tol) or np.any(x > w - r + tol):
            return False, 0.0
        if np.any(y < r - tol) or np.any(y > h - r + tol):
            return False, 0.0
        dx = x[_IU[0]] - x[_IU[1]]
        dy = y[_IU[0]] - y[_IU[1]]
        d2 = dx * dx + dy * dy
        r_sum = r[_IU[0]] + r[_IU[1]]
        if np.any(d2 < r_sum * r_sum - tol):
            return False, 0.0
        return True, float(np.sum(r))

    def force_resolve(x, y, r, w, h, max_iter=500, dt=0.003):
        """Resolve overlaps via force-directed relaxation, clipping to bounds."""
        x = x.copy()
        y = y.copy()
        n = N
        for it in range(max_iter):
            fx = np.zeros(n)
            fy = np.zeros(n)

            # Wall forces
            fx += np.where(x < r, (r - x) * 100, 0.0)
            fx -= np.where(x > w - r, (x - (w - r)) * 100, 0.0)
            fy += np.where(y < r, (r - y) * 100, 0.0)
            fy -= np.where(y > h - r, (y - (h - r)) * 100, 0.0)

            # Pairwise forces
            dx = x[_IU[0]] - x[_IU[1]]
            dy = y[_IU[0]] - y[_IU[1]]
            d = np.sqrt(dx * dx + dy * dy)
            r_sum = r[_IU[0]] + r[_IU[1]]
            overlap = np.maximum(r_sum - d, 0.0)
            max_ov = np.max(overlap) if len(overlap) > 0 else 0.0
            if max_ov < 1e-10:
                break

            mask = overlap > 1e-12
            if np.any(mask):
                idx_i = _IU[0][mask]
                idx_j = _IU[1][mask]
                d_m = np.maximum(d[mask], 1e-12)
                ux = dx[mask] / d_m
                uy = dy[mask] / d_m
                f = overlap[mask] * 30
                np.add.at(fx, idx_i, ux * f)
                np.add.at(fy, idx_i, uy * f)
                np.add.at(fx, idx_j, -ux * f)
                np.add.at(fy, idx_j, -uy * f)

            x += dt * fx
            y += dt * fy
            x = np.clip(x, r, w - r)
            y = np.clip(y, r, h - r)

        return x, y

    def expand_radii(x, y, r, w, h, max_rounds=80):
        """Greedy radius expansion: for each circle, compute max feasible radius."""
        x = x.copy()
        y = y.copy()
        r = r.copy()
        n = N
        for _ in range(max_rounds):
            improved = False
            for i in range(n):
                r_wall = min(x[i], w - x[i], y[i], h - y[i])
                dx = x[i] - x
                dy = y[i] - y
                d = np.sqrt(dx * dx + dy * dy)
                d[i] = np.inf
                r_allowed = d - r
                r_max = min(r_wall, np.min(r_allowed))
                r_max = max(0.0, r_max)
                if r_max > r[i] + 1e-9:
                    r[i] = r_max
                    improved = True
            if not improved:
                break
        return x, y, r

    def slsqp_hard(x0, y0, r0, w0, h0, time_budget=25):
        """
        SLSQP with hard constraints using radial coordinates.
        Variables: u (n), v (n), r (n), w (1) where x = r + u, y = r + v.
        Constraints:
          - u >= 0, v >= 0 (wall lower)
          - u <= w - 2*r (wall right) -- this is coupled, so use x-based
          Actually, let's keep x,y,r,w as variables but use hard constraints.
        """
        # Variables: x(0..n-1), y(n..2n-1), r(2n..3n-1), w(3n)
        z0 = np.zeros(3 * N + 1)
        z0[0:N] = x0
        z0[N:2*N] = y0
        z0[2*N:3*N] = r0
        z0[3*N] = w0

        # Bounds
        bounds = []
        for i in range(N):
            bounds.append((0.001, 2.0))  # x
        for i in range(N):
            bounds.append((0.001, 2.0))  # y
        for i in range(N):
            bounds.append((0.001, 1.0))  # r
        bounds.append((0.3, 1.7))  # w

        # Constraints (hard inequality: c >= 0)
        cons = []

        # Wall constraints: x_i >= r_i, x_i + r_i <= w, y_i >= r_i, y_i + r_i <= h
        # h = 2 - w
        # x_i - r_i >= 0
        def wall_left(z):
            return z[0:N] - z[2*N:3*N]
        cons.append({'type': 'ineq', 'fun': wall_left})

        # x_i + r_i - w <= 0  =>  w - x_i - r_i >= 0
        def wall_right(z):
            w = z[3*N]
            return w - z[0:N] - z[2*N:3*N]
        cons.append({'type': 'ineq', 'fun': wall_right})

        # y_i - r_i >= 0
        def wall_bottom(z):
            return z[N:2*N] - z[2*N:3*N]
        cons.append({'type': 'ineq', 'fun': wall_bottom})

        # y_i + r_i - h <= 0  =>  h - y_i - r_i >= 0  =>  (2-w) - y_i - r_i >= 0
        def wall_top(z):
            w = z[3*N]
            h = 2.0 - w
            return h - z[N:2*N] - z[2*N:3*N]
        cons.append({'type': 'ineq', 'fun': wall_top})

        # Circle-circle: (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
        # => (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
        def circle_circle(z):
            x = z[0:N]
            y = z[N:2*N]
            r = z[2*N:3*N]
            dx = x[_IU[0]] - x[_IU[1]]
            dy = y[_IU[0]] - y[_IU[1]]
            d2 = dx * dx + dy * dy
            rs = r[_IU[0]] + r[_IU[1]]
            return d2 - rs * rs
        cons.append({'type': 'ineq', 'fun': circle_circle})

        def objective(z):
            r = z[2*N:3*N]
            return -np.sum(r)

        maxiter = int(min(200, time_budget * 15))
        try:
            res = minimize(objective, z0, method='SLSQP', bounds=bounds,
                           constraints=cons,
                           options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 20})
            z_opt = res.x
            x = z_opt[0:N]
            y = z_opt[N:2*N]
            r = z_opt[2*N:3*N]
            w = z_opt[3*N]
            h = 2.0 - w
            # Slight shrink for safety
            r_safe = r * 0.999
            valid, val = check_validity(x, y, r_safe, w, h)
            if valid:
                return x, y, r_safe, w, h, val
            # Try without shrink
            valid, val = check_validity(x, y, r, w, h)
            if valid:
                return x, y, r, w, h, val
            return None
        except Exception:
            return None

    def local_search(x, y, r, w, h, time_budget=8):
        """Perturbation-based local search to escape local minima."""
        x = x.copy()
        y = y.copy()
        r = r.copy()
        n = N
        end_time = time.time() + time_budget
        rng = np.random.default_rng(42)
        
        for _ in range(500):
            if time.time() > end_time:
                break
            improved = False
            # Try moving a random circle slightly
            for _ in range(20):
                i = rng.integers(n)
                dx = rng.uniform(-0.01, 0.01)
                dy = rng.uniform(-0.01, 0.01)
                new_x = x[i] + dx
                new_y = y[i] + dy
                if new_x < r[i] or new_x > w - r[i]:
                    continue
                if new_y < r[i] or new_y > h - r[i]:
                    continue
                # Check validity
                ddx = x - new_x
                ddy = y - new_y
                d2 = ddx * ddx + ddy * ddy
                r_sum = r + r[i]
                d2[i] = np.inf
                r_sum[i] = 0
                if np.all(d2 >= r_sum * r_sum - 1e-10):
                    x[i] = new_x
                    y[i] = new_y
                    improved = True
                    break
            if improved:
                # Try to expand radii after move
                x, y, r = expand_radii(x, y, r, w, h, max_rounds=10)
            else:
                # Try expanding a random circle
                i = rng.integers(n)
                r_wall = min(x[i], w - x[i], y[i], h - y[i])
                dx_arr = x[i] - x
                dy_arr = y[i] - y
                d = np.sqrt(dx_arr * dx_arr + dy_arr * dy_arr)
                d[i] = np.inf
                r_allowed = d - r
                r_max = min(r_wall, np.min(r_allowed))
                if r_max > r[i] + 1e-9:
                    r[i] = r_max
                    improved = True

        return x, y, r

    def get_initial_guesses():
        guesses = []
        side = int(np.ceil(np.sqrt(N)))

        aspect_ratios = [1.0, 0.95, 1.05, 0.9, 1.1, 0.85, 1.15, 0.8, 1.2, 0.75, 1.25, 0.7, 1.3, 0.65, 1.4, 0.6, 1.5, 0.55, 1.6, 0.5]
        for w0 in aspect_ratios:
            h0 = 2.0 - w0
            if h0 < 0.3:
                continue
            x = np.zeros(N)
            y = np.zeros(N)
            r = np.zeros(N)
            for i in range(N):
                row, col = divmod(i, side)
                cx = (col + 0.5) / side * w0
                cy = (row + 0.5) / side * h0
                r0 = min(w0 / side, h0 / side) * 0.45
                x[i] = cx
                y[i] = cy
                r[i] = r0
            guesses.append((x, y, r, w0, h0))

        # Random starts
        rng = np.random.default_rng(99)
        for _ in range(8):
            w0 = rng.uniform(0.6, 1.4)
            h0 = 2.0 - w0
            x = np.zeros(N)
            y = np.zeros(N)
            r = np.zeros(N)
            for i in range(N):
                r0 = rng.uniform(0.04, 0.10)
                x[i] = rng.uniform(r0, w0 - r0)
                y[i] = rng.uniform(r0, h0 - r0)
                r[i] = r0
            guesses.append((x, y, r, w0, h0))

        return guesses

    guesses = get_initial_guesses()
    guesses.sort(key=lambda g: abs(g[3] - 1.0))

    for i, (x0, y0, r0, w0, h0) in enumerate(guesses):
        if time.time() > deadline - 30:
            break

        # Phase 1: Force resolve
        x, y = force_resolve(x0, y0, r0, w0, h0, max_iter=300)
        valid, val = check_validity(x, y, r0, w0, h0)
        if not valid:
            # Shrink radii slightly to get valid
            r0_safe = r0 * 0.9
            valid, val = check_validity(x, y, r0_safe, w0, h0)
            if not valid:
                continue
            r0 = r0_safe

        # Phase 2: Hard-constraint SLSQP
        remaining = deadline - (time.time() - start_time)
        time_budget = min(25, remaining - 25)
        if time_budget > 8:
            result = slsqp_hard(x, y, r0, w0, h0, time_budget)
            if result is not None:
                x, y, r, w, h, val = result
            else:
                r = r0.copy()
                w, h = w0, h0
        else:
            r = r0.copy()
            w, h = w0, h0

        # Phase 3: Expand radii
        x, y, r = expand_radii(x, y, r, w, h, max_rounds=60)

        # Phase 4: Local search
        remaining = deadline - (time.time() - start_time)
        if remaining > 15:
            x, y, r = local_search(x, y, r, w, h, time_budget=min(8, remaining - 15))
            x, y, r = expand_radii(x, y, r, w, h, max_rounds=40)

        valid, val = check_validity(x, y, r, w, h)
        if valid and val > best_value:
            best_value = val
            best_result = (x.copy(), y.copy(), r.copy(), w, h)

    # Final polish
    if best_result is not None and time.time() < deadline - 10:
        x, y, r, w, h = best_result
        x, y, r = expand_radii(x, y, r, w, h, max_rounds=80)
        remaining = deadline - (time.time() - start_time)
        if remaining > 5:
            x, y, r = local_search(x, y, r, w, h, time_budget=min(5, remaining - 5))
            x, y, r = expand_radii(x, y, r, w, h, max_rounds=60)
        valid, val = check_validity(x, y, r, w, h)
        if valid and val > best_value:
            best_value = val
            best_result = (x.copy(), y.copy(), r.copy(), w, h)

    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(N):
            row, col = divmod(i, side)
            cx = (col + 0.5) / side
            cy = (row + 0.5) / side
            r0 = 0.5 / side - 0.005
            cells.append((cx, cy, r0))
        return np.array(cells)

    x, y, r, w, h = best_result
    return np.column_stack((x, y, r))
