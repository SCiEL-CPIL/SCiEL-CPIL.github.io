import numpy as np
from scipy.optimize import minimize
import time

N = 21


def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s kill

    best_r = 0.0
    best_x = None
    best_y = None
    best_W = 1.0
    best_H = 1.0

    # --- Validation ---
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

    # --- Force-directed relaxation (unified: positions, r, W all free) ---
    def force_relax(x, y, r, W, max_iter=200, dt=0.003):
        x = x.astype(float).copy()
        y = y.astype(float).copy()
        r = float(r)
        W = float(W)
        H = 2.0 - W
        for _ in range(max_iter):
            fx = np.zeros(N)
            fy = np.zeros(N)
            # Wall constraints
            for i in range(N):
                if x[i] - r < 0:
                    fx[i] += (x[i] - r) * 30
                if x[i] + r > W:
                    fx[i] -= (x[i] + r - W) * 30
                if y[i] - r < 0:
                    fy[i] += (y[i] - r) * 30
                if y[i] + r > H:
                    fy[i] -= (y[i] + r - H) * 30
            # Pairwise repulsion
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx * dx + dy * dy)
                    if d < 2 * r and d > 1e-12:
                        overlap = 2 * r - d
                        fx[i] += (dx / d) * overlap * 15
                        fy[i] += (dy / d) * overlap * 15
                        fx[j] -= (dx / d) * overlap * 15
                        fy[j] -= (dy / d) * overlap * 15
            x += dt * fx
            y += dt * fy
            x = np.clip(x, max(r, 1e-6), max(W - r, 1e-6))
            y = np.clip(y, max(r, 1e-6), max(H - r, 1e-6))
        return x, y

    # --- Polish: greedy radius growth ---
    def polish(x, y, r, W, H, time_budget):
        x = x.astype(float).copy()
        y = y.astype(float).copy()
        r = float(r)
        t0 = time.time()
        improved = True
        while improved and (time.time() - t0) < time_budget:
            improved = False
            max_bump = min(
                min(x[i] - r for i in range(N)),
                min(W - r - x[i] for i in range(N)),
                min(y[i] - r for i in range(N)),
                min(H - r - y[i] for i in range(N))
            )
            for i in range(N):
                for j in range(i + 1, N):
                    d = np.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2)
                    max_bump = min(max_bump, (d - 2 * r) / 2.0)
            if max_bump <= 1e-10:
                break
            for frac in (0.95, 0.7, 0.4, 0.2, 0.05):
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

    # --- SLSQP with free aspect ratio ---
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
        cons.append(r - 0.005)
        cons.append(W - 0.3)
        cons.append(H - 0.3)
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                cons.append(dx * dx + dy * dy - (2 * r) ** 2)
        return np.concatenate(cons)

    constraints = [{'type': 'ineq', 'fun': con_bounds}]

    def run_slsqp(x0, y0, r0, W0, maxiter):
        H0 = 2.0 - W0
        v0 = np.concatenate([x0, y0, [r0, W0]])
        res = minimize(
            objective, v0, method='SLSQP',
            bounds=[(0.01, 1.99)] * N + [(0.01, 1.99)] * N + [(0.01, 0.8)] + [(0.35, 1.65)],
            constraints=constraints,
            options={'maxiter': maxiter, 'ftol': 1e-10, 'maxls': 80}
        )
        x, y, r, W, H = unpack(res.x)
        return x, y, r, W, H

    # --- Initial guesses ---
    def make_starts():
        starts = []
        # Hexagonal-style dense grid for various aspect ratios
        aspect_ratios = [1.0, 0.9, 1.1, 0.8, 1.2, 0.85, 1.15, 0.75, 1.3,
                         0.7, 1.4, 0.65, 1.5, 0.6, 1.7, 0.55, 1.9, 0.5, 2.2]
        for W0 in aspect_ratios:
            H0 = 2.0 - W0
            if H0 < 0.3 or W0 < 0.3:
                continue
            # Hexagonal packing pattern
            x0 = np.empty(N)
            y0 = np.empty(N)
            n_per_row = int(np.ceil(np.sqrt(N * H0 / W0)))
            if n_per_row < 2:
                n_per_row = 2
            rows = int(np.ceil(N / n_per_row))
            spacing_x = W0 / (n_per_row + 0.5)
            spacing_y = H0 / (rows + 0.5)
            r_est = min(spacing_x, spacing_y) * 0.48
            idx = 0
            for row in range(rows):
                for col in range(n_per_row):
                    if idx >= N:
                        break
                    x0[idx] = (col + 0.5) * spacing_x
                    y0[idx] = (row + 0.5) * spacing_y
                    if row % 2 == 1:
                        x0[idx] += spacing_x * 0.3
                    idx += 1
            # Fill remaining
            while idx < N:
                x0[idx] = W0 * 0.5
                y0[idx] = H0 * 0.5
                idx += 1
            r0 = min(r_est, min(W0, H0) * 0.35) * 0.5
            starts.append((x0, y0, r0, W0))
        # Random starts
        rng = np.random.default_rng(42)
        for _ in range(8):
            W0 = rng.uniform(0.6, 1.5)
            H0 = 2.0 - W0
            if H0 < 0.3:
                continue
            x0 = rng.uniform(0.05, W0 - 0.05, N)
            y0 = rng.uniform(0.05, H0 - 0.05, N)
            r0 = rng.uniform(0.04, 0.09)
            starts.append((x0, y0, r0, W0))
        return starts

    starts = make_starts()

    for i, (x0, y0, r0, W0) in enumerate(starts):
        if time.time() > deadline - 10:
            break
        H0 = 2.0 - W0
        try:
            # Pre-relax
            xr, yr = force_relax(x0, y0, r0, W0, max_iter=80)

            remaining = deadline - time.time()
            maxiter = int(min(200, max(50, remaining * 20)))
            x, y, r, W, H = run_slsqp(xr, yr, r0, W0, maxiter)

            if check_valid(x, y, r, W, H):
                # Polish
                if time.time() < deadline - 8:
                    xp, yp, rp = polish(x, y, r, W, H, time_budget=2.0)
                    if check_valid(xp, yp, rp, W, H) and rp > r:
                        x, y, r = xp, yp, rp
                if r > best_r:
                    best_r = r
                    best_x = x.copy()
                    best_y = y.copy()
                    best_W = W
                    best_H = H
            else:
                # Try to repair by shrinking
                for shrink in (0.95, 0.9, 0.85, 0.8, 0.7):
                    rs = r * shrink
                    if check_valid(x, y, rs, W, H) and rs > best_r:
                        best_r = rs
                        best_x = x.copy()
                        best_y = y.copy()
                        best_W = W
                        best_H = H
                        break
        except Exception:
            continue

    # Final polish
    if best_x is not None and time.time() < deadline - 5:
        try:
            xp, yp, rp = polish(best_x, best_y, best_r, best_W, best_H,
                                time_budget=min(8.0, deadline - time.time() - 3))
            if check_valid(xp, yp, rp, best_W, best_H) and rp > best_r:
                best_r, best_x, best_y = rp, xp, yp
        except Exception:
            pass

    # Second round: re-optimize from best solution with perturbation
    if best_x is not None and time.time() < deadline - 20:
        try:
            rng2 = np.random.default_rng(123)
            for attempt in range(3):
                if time.time() > deadline - 10:
                    break
                # Perturb best solution
                x_p = best_x + rng2.normal(0, best_r * 0.1, N)
                y_p = best_y + rng2.normal(0, best_r * 0.1, N)
                W_p = best_W + rng2.uniform(-0.1, 0.1)
                H_p = 2.0 - W_p
                if H_p < 0.3 or W_p < 0.3:
                    continue
                x_p = np.clip(x_p, best_r * 0.5, W_p - best_r * 0.5)
                y_p = np.clip(y_p, best_r * 0.5, H_p - best_r * 0.5)
                r_p = best_r * 0.85
                xr2, yr2 = force_relax(x_p, y_p, r_p, W_p, max_iter=60)
                remaining = deadline - time.time()
                maxiter2 = int(min(150, max(40, remaining * 15)))
                x2, y2, r2, W2, H2 = run_slsqp(xr2, yr2, r_p, W_p, maxiter2)
                if check_valid(x2, y2, r2, W2, H2):
                    if time.time() < deadline - 5:
                        xp2, yp2, rp2 = polish(x2, y2, r2, W2, H2, time_budget=1.5)
                        if check_valid(xp2, yp2, rp2, W2, H2) and rp2 > r2:
                            x2, y2, r2 = xp2, yp2, rp2
                    if r2 > best_r:
                        best_r, best_x, best_y, best_W, best_H = r2, x2.copy(), y2.copy(), W2, H2
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
