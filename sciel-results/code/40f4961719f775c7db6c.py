"""
Circle packing in perimeter-4 rectangle: 21 circles.
Structural change: use a force-directed / gradient-free local search with
  - a compact hexagonal grid initialization
  - iterative radius growth with position relaxation via scipy.optimize.minimize
  - multiple aspect-ratio restarts
  - final polish with Nelder-Mead on positions only
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 21


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    def is_valid(x, y, r, w, tol=1e-9):
        h = 2.0 - w
        if w <= tol or h <= tol:
            return False
        for i in range(N):
            if r[i] <= tol:
                return False
            if x[i] < r[i] - tol or x[i] > w - r[i] + tol:
                return False
            if y[i] < r[i] - tol or y[i] > h - r[i] + tol:
                return False
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                if dx * dx + dy * dy < (r[i] + r[j]) ** 2 - tol:
                    return False
        return True

    def sum_radii(r):
        return float(np.sum(r))

    def hex_init(w0, n=N):
        """Generate hexagonal lattice initial positions with uniform radius."""
        h0 = 2.0 - w0
        # Estimate radius: area = w*h, hex packing efficiency ~0.9069
        area = w0 * h0
        r_est = np.sqrt(area / (n * np.pi / 0.9069))
        r_est = min(r_est, 0.14)

        x = np.zeros(n)
        y = np.zeros(n)
        r = np.full(n, r_est)

        dx = 2.0 * r_est * 1.001  # tiny gap
        dy = np.sqrt(3) * r_est * 1.001
        row = 0
        col = 0
        idx = 0
        while idx < n:
            cx = dx * (col + 0.5 * (row % 2)) + r_est
            cy = dy * row + r_est
            if cx > w0 - r_est:
                row += 1
                col = 0
                cx = dx * (col + 0.5 * (row % 2)) + r_est
                cy = dy * row + r_est
            if cy > h0 - r_est:
                break
            x[idx] = cx
            y[idx] = cy
            idx += 1
            col += 1

        # If not enough placed, fill remaining at center with smaller radius
        for i in range(idx, n):
            x[i] = w0 / 2.0
            y[i] = h0 / 2.0
            r[i] = r_est * 0.7

        return x, y, r

    def relax_positions(x, y, r, w, max_iter=200, ftol=1e-12):
        """Relax positions to minimize overlap and wall violations."""
        h = 2.0 - w

        def obj(z):
            xz = z[:N]
            yz = z[N:2 * N]
            pen = 0.0
            # Wall penalties
            for i in range(N):
                if xz[i] < r[i]:
                    pen += (r[i] - xz[i]) ** 2
                if xz[i] > w - r[i]:
                    pen += (xz[i] - (w - r[i])) ** 2
                if yz[i] < r[i]:
                    pen += (r[i] - yz[i]) ** 2
                if yz[i] > h - r[i]:
                    pen += (yz[i] - (h - r[i])) ** 2
            # Overlap penalties
            for i in range(N):
                for j in range(i + 1, N):
                    dx = xz[i] - xz[j]
                    dy = yz[i] - yz[j]
                    d = np.sqrt(dx * dx + dy * dy) + 1e-12
                    ov = r[i] + r[j] - d
                    if ov > 0:
                        pen += ov ** 2
            return pen

        z0 = np.concatenate([x, y])
        bounds = [(r[i], w - r[i]) for i in range(N)] + \
                 [(r[i], h - r[i]) for i in range(N)]

        try:
            res = minimize(obj, z0, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': max_iter, 'ftol': ftol, 'maxls': 40})
            return res.x[:N], res.x[N:2 * N]
        except Exception:
            return x.copy(), y.copy()

    def try_grow(x, y, r, w, time_left):
        """Iteratively grow radii and relax positions."""
        h = 2.0 - w
        improved = True
        while improved and (time.time() - start_time) < deadline - 5:
            improved = False
            for i in range(N):
                if (time.time() - start_time) > deadline - 5:
                    break
                # Try to grow radius i slightly
                for _ in range(30):
                    r_new = r.copy()
                    r_new[i] *= 1.002
                    # Check if it fits
                    x_t, y_t = relax_positions(x, y, r_new, w, max_iter=50, ftol=1e-10)
                    if is_valid(x_t, y_t, r_new, w):
                        x, y, r = x_t, y_t, r_new
                        improved = True
                    else:
                        break
        return x, y, r, w

    def polish(x, y, r, w, time_left):
        """Nelder-Mead polish on positions only."""
        h = 2.0 - w

        def obj(z):
            xz = z[:N]
            yz = z[N:2 * N]
            pen = 0.0
            for i in range(N):
                if xz[i] < r[i]:
                    pen += (r[i] - xz[i]) ** 2
                if xz[i] > w - r[i]:
                    pen += (xz[i] - (w - r[i])) ** 2
                if yz[i] < r[i]:
                    pen += (r[i] - yz[i]) ** 2
                if yz[i] > h - r[i]:
                    pen += (yz[i] - (h - r[i])) ** 2
            for i in range(N):
                for j in range(i + 1, N):
                    dx = xz[i] - xz[j]
                    dy = yz[i] - yz[j]
                    d = np.sqrt(dx * dx + dy * dy) + 1e-12
                    ov = r[i] + r[j] - d
                    if ov > 0:
                        pen += ov ** 2
            return pen

        z0 = np.concatenate([x, y])
        try:
            res = minimize(obj, z0, method='Nelder-Mead',
                           options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-12})
            x_t, y_t = res.x[:N], res.x[N:2 * N]
            if is_valid(x_t, y_t, r, w):
                return x_t, y_t, r, w
        except Exception:
            pass
        return x, y, r, w

    best_value = 0.0
    best_result = None
    rng = np.random.default_rng(42)

    # Phase 1: Try multiple aspect ratios with hex initialization
    aspect_ratios = [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.7, 0.55, 2.0, 0.5, 1.1, 0.9, 1.25, 0.8, 1.4, 0.7, 1.6, 0.65]
    for w0 in aspect_ratios:
        if time.time() > deadline - 20:
            break
        h0 = 2.0 - w0
        if h0 < 0.3:
            continue

        x0, y0, r0 = hex_init(w0)
        # Relax positions
        x_t, y_t = relax_positions(x0, y0, r0, w0, max_iter=200, ftol=1e-12)
        if not is_valid(x_t, y_t, r0, w0):
            continue
        val = sum_radii(r0)
        if val > best_value:
            best_value = val
            best_result = (x_t.copy(), y_t.copy(), r0.copy(), w0)

    # Phase 2: Grow radii from best
    if best_result is not None:
        x_cur, y_cur, r_cur, w_cur = best_result
        x_cur, y_cur, r_cur, w_cur = try_grow(x_cur, y_cur, r_cur, w_cur, deadline - (time.time() - start_time))
        val = sum_radii(r_cur) if is_valid(x_cur, y_cur, r_cur, w_cur) else 0.0
        if val > best_value:
            best_value = val
            best_result = (x_cur.copy(), y_cur.copy(), r_cur.copy(), w_cur)

    # Phase 3: Polish
    if best_result is not None and time.time() < deadline - 10:
        x_cur, y_cur, r_cur, w_cur = best_result
        x_cur, y_cur, r_cur, w_cur = polish(x_cur, y_cur, r_cur, w_cur, deadline - (time.time() - start_time))
        val = sum_radii(r_cur) if is_valid(x_cur, y_cur, r_cur, w_cur) else 0.0
        if val > best_value:
            best_value = val
            best_result = (x_cur.copy(), y_cur.copy(), r_cur.copy(), w_cur)

    # Phase 4: Additional random restarts if time permits
    if time.time() < deadline - 15:
        for _ in range(8):
            if time.time() > deadline - 10:
                break
            w0 = rng.uniform(0.55, 1.45)
            h0 = 2.0 - w0
            if h0 < 0.3:
                continue
            x0, y0, r0 = hex_init(w0)
            # Perturb
            x0 += rng.normal(0, 0.01, N)
            y0 += rng.normal(0, 0.01, N)
            x_t, y_t = relax_positions(x0, y0, r0, w0, max_iter=100, ftol=1e-10)
            if not is_valid(x_t, y_t, r0, w0):
                continue
            val = sum_radii(r0)
            if val > best_value:
                # Try to grow
                x_g, y_g, r_g, w_g = try_grow(x_t, y_t, r0, w0, deadline - (time.time() - start_time))
                val_g = sum_radii(r_g) if is_valid(x_g, y_g, r_g, w_g) else val
                if val_g > best_value:
                    best_value = val_g
                    best_result = (x_g.copy(), y_g.copy(), r_g.copy(), w_g)

    # Phase 5: Final polish
    if best_result is not None and time.time() < deadline - 5:
        x_cur, y_cur, r_cur, w_cur = best_result
        x_cur, y_cur, r_cur, w_cur = polish(x_cur, y_cur, r_cur, w_cur, deadline - (time.time() - start_time))
        if is_valid(x_cur, y_cur, r_cur, w_cur):
            best_result = (x_cur.copy(), y_cur.copy(), r_cur.copy(), w_cur)

    if best_result is None:
        # Fallback: simple grid
        cols = 5
        rows = 5
        cells = []
        for i in range(N):
            row, col = divmod(i, cols)
            cx = (col + 0.5) / cols
            cy = (row + 0.5) / rows
            r0 = min(1.0 / cols, 1.0 / rows) * 0.45
            cells.append((cx, cy, r0))
        return np.array(cells)

    x_final, y_final, r_final, w_final = best_result
    # Safety shrink
    for _ in range(5):
        r_trial = r_final * (1 - 1e-10)
        if is_valid(x_final, y_final, r_trial, w_final):
            r_final = r_trial
        else:
            break

    return np.column_stack((x_final, y_final, r_final))
