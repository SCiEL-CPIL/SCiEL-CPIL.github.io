import numpy as np
from scipy.optimize import minimize
import time

N = 21


def construct():
    start_time = time.time()
    deadline = start_time + 138.0  # safety margin before 150s kill

    best_r = 0.0
    best_x = None
    best_y = None
    best_W = 1.0
    best_H = 1.0

    # --- Parameterization: equal radii r, free aspect ratio W, H = 2 - W ---
    def unpack(v):
        x = v[0:N]
        y = v[N:2 * N]
        r = v[2 * N]
        W = v[2 * N + 1]
        H = 2.0 - W
        return x, y, r, W, H

    def objective(v):
        x, y, r, W, H = unpack(v)
        return -r

    def con_bounds(v):
        x, y, r, W, H = unpack(v)
        cons = []
        cons.append(x - r)
        cons.append((W - r) - x)
        cons.append(y - r)
        cons.append((H - r) - y)
        cons.append(r)
        cons.append(W)
        cons.append(H)
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx * dx + dy * dy
                min_dist_sq = (2 * r) ** 2
                cons.append(dist_sq - min_dist_sq)
        return np.concatenate(cons)

    constraints = [{'type': 'ineq', 'fun': con_bounds}]

    def check_valid(x, y, r, W, H, tol=1e-7):
        if r <= 0 or W <= 0.05 or H <= 0.05:
            return False
        for k in range(N):
            if x[k] < r - tol or x[k] > W - r + tol:
                return False
            if y[k] < r - tol or y[k] > H - r + tol:
                return False
        for k in range(N):
            for l in range(k + 1, N):
                if (x[k] - x[l]) ** 2 + (y[k] - y[l]) ** 2 < (2 * r) ** 2 - tol:
                    return False
        return True

    # --- Force-directed pre-relaxation to resolve initial overlaps ---
    def force_relax(x, y, r, W, H, max_iter=120):
        x = x.astype(float).copy()
        y = y.astype(float).copy()
        dt = 0.004
        for _ in range(max_iter):
            fx = np.zeros(N)
            fy = np.zeros(N)
            for i in range(N):
                if x[i] - r < 0:
                    fx[i] += (x[i] - r) * 25
                if x[i] + r > W:
                    fx[i] -= (x[i] + r - W) * 25
                if y[i] - r < 0:
                    fy[i] += (y[i] - r) * 25
                if y[i] + r > H:
                    fy[i] -= (y[i] + r - H) * 25
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx * dx + dy * dy)
                    if d < 2 * r and d > 1e-12:
                        overlap = 2 * r - d
                        fx[i] += (dx / d) * overlap * 12
                        fy[i] += (dy / d) * overlap * 12
                        fx[j] -= (dx / d) * overlap * 12
                        fy[j] -= (dy / d) * overlap * 12
            x += dt * fx
            y += dt * fy
            x = np.clip(x, r, W - r)
            y = np.clip(y, r, H - r)
        return x, y

    # --- Post-polish: greedy radius growth on a valid packing ---
    def polish(x, y, r, W, H, time_budget):
        x = x.astype(float).copy()
        y = y.astype(float).copy()
        r = float(r)
        t0 = time.time()
        improved = True
        while improved and (time.time() - t0) < time_budget:
            improved = False
            # find max possible uniform radius bump
            max_bump = 0.0
            for i in range(N):
                max_bump = min(max_bump, x[i] - r, W - r - x[i], y[i] - r, H - r - y[i])
            # pairwise: bump limited by min over pairs of (dist - 2r)/2
            for i in range(N):
                for j in range(i + 1, N):
                    d = np.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2)
                    max_bump = min(max_bump, (d - 2 * r) / 2.0)
            if max_bump <= 1e-9:
                break
            # try a fraction of the bump
            for frac in (0.9, 0.5, 0.25, 0.1):
                new_r = r + frac * max_bump
                if new_r <= r:
                    continue
                if check_valid(x, y, new_r, W, H, tol=1e-8):
                    r = new_r
                    improved = True
                    break
            if not improved:
                break
        return x, y, r

    # --- Initial guesses: dense grid of aspect ratios ---
    def make_starts():
        starts = []
        aspect_ratios = [1.0, 0.9, 1.1, 0.8, 1.25, 0.75, 1.4, 0.7, 1.6, 0.65,
                         1.8, 0.6, 2.0, 0.55, 2.2, 0.5, 0.45, 2.5]
        side = int(np.ceil(np.sqrt(N)))
        for W0 in aspect_ratios:
            H0 = 2.0 - W0
            if H0 < 0.25 or W0 < 0.25:
                continue
            x0 = np.empty(N)
            y0 = np.empty(N)
            for i in range(N):
                row, col = divmod(i, side)
                x0[i] = (col + 0.5) / side * W0
                y0[i] = (row + 0.5) / side * H0
            r0 = min(W0, H0) / side * 0.45
            starts.append((x0, y0, r0, W0))
        # a few random starts
        rng = np.random.default_rng(7)
        for _ in range(6):
            W0 = rng.uniform(0.6, 1.4)
            H0 = 2.0 - W0
            x0 = rng.uniform(0.1, W0 - 0.1, N)
            y0 = rng.uniform(0.1, H0 - 0.1, N)
            r0 = rng.uniform(0.04, 0.1)
            starts.append((x0, y0, r0, W0))
        return starts

    starts = make_starts()

    for i, (x0, y0, r0, W0) in enumerate(starts):
        if time.time() > deadline - 5:
            break
        H0 = 2.0 - W0
        try:
            # pre-relax
            xr, yr = force_relax(x0, y0, r0, W0, H0, max_iter=100)
            v0 = np.concatenate([xr, yr, [r0, W0]])

            remaining = deadline - time.time()
            maxiter = int(min(250, max(60, remaining * 25)))
            res = minimize(objective, v0, method='SLSQP',
                           bounds=[(0, 2)] * N + [(0, 2)] * N + [(0.01, 1.0)] + [(0.4, 1.6)],
                           constraints=constraints,
                           options={'maxiter': maxiter, 'ftol': 1e-9, 'maxls': 60})

            x, y, r, W, H = unpack(res.x)
            if check_valid(x, y, r, W, H):
                # polish
                if time.time() < deadline - 8:
                    xp, yp, rp = polish(x, y, r, W, H, time_budget=2.0)
                    if check_valid(xp, yp, rp, W, H) and rp > r:
                        x, y, r = xp, yp, rp
                if r > best_r:
                    best_r = r
                    best_x = x
                    best_y = y
                    best_W = W
                    best_H = H
            else:
                # repair: shrink radius
                for shrink in (0.95, 0.9, 0.85, 0.8):
                    rs = r * shrink
                    if check_valid(x, y, rs, W, H) and rs > best_r:
                        best_r = rs
                        best_x = x
                        best_y = y
                        best_W = W
                        best_H = H
                        break
        except Exception:
            continue

    # Final polish on the best result if there is time
    if best_x is not None and time.time() < deadline - 5:
        try:
            xp, yp, rp = polish(best_x, best_y, best_r, best_W, best_H,
                                time_budget=deadline - time.time() - 3)
            if check_valid(xp, yp, rp, best_W, best_H) and rp > best_r:
                best_r, best_x, best_y = rp, xp, yp
        except Exception:
            pass

    if best_x is not None:
        result = np.column_stack((best_x, best_y, np.full(N, best_r)))
        return result
    else:
        # Fallback: simple grid in unit square
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
