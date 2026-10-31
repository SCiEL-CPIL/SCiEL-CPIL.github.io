import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import cKDTree

N = 21
_IU = np.triu_indices(N, k=1)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def elapsed():
        return time.time() - start_time

    def time_left():
        return deadline - elapsed()

    # --- Validity check ---
    def check_validity(x, y, r, w, tol=1e-7):
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
                rs = r[i] + r[j]
                if d2 < rs * rs - tol:
                    return False, 0.0
        return True, float(np.sum(r))

    # --- Pebble inflation: greedy constructive packing ---
    def pebble_inflate(w, h, n_circles=N, seed=42):
        """
        Greedily place circles at max radius in free space.
        Returns (x, y, r, w) or None.
        """
        rng = np.random.default_rng(seed)
        x = np.zeros(n_circles)
        y = np.zeros(n_circles)
        r = np.zeros(n_circles)

        # Start with a grid of small circles
        side = int(np.ceil(np.sqrt(n_circles)))
        for k in range(n_circles):
            row, col = divmod(k, side)
            x[k] = (col + 0.5) * w / side
            y[k] = (row + 0.5) * h / side
            r[k] = min(w / side, h / side) * 0.4

        # Iteratively inflate
        max_rounds = 50
        for rnd in range(max_rounds):
            if time_left() < 5:
                break
            # Shuffle order for diversity
            order = rng.permutation(n_circles)
            improved = False
            for k in order:
                # Max radius from walls
                max_r = min(x[k], w - x[k], y[k], h - y[k])
                # Constraints with other circles
                for j in range(n_circles):
                    if j == k:
                        continue
                    dx = x[k] - x[j]
                    dy = y[k] - y[j]
                    d = np.sqrt(dx * dx + dy * dy)
                    c = d - r[j]
                    if c < max_r:
                        max_r = c
                if max_r > r[k] + 1e-10:
                    r[k] = max_r - 1e-10
                    improved = True
            if not improved:
                break

        # Try to reposition circles to allow further growth
        for rnd in range(20):
            if time_left() < 5:
                break
            # For each circle, try moving it to a position that allows larger radius
            for k in range(n_circles):
                # Sample candidate positions
                best_r = r[k]
                best_x, best_y = x[k], y[k]
                n_samples = 20
                for _ in range(n_samples):
                    cx = rng.uniform(r[k], w - r[k])
                    cy = rng.uniform(r[k], h - r[k])
                    # Max radius at this position
                    max_r = min(cx, w - cx, cy, h - cy)
                    for j in range(n_circles):
                        if j == k:
                            continue
                        dx = cx - x[j]
                        dy = cy - y[j]
                        d = np.sqrt(dx * dx + dy * dy)
                        c = d - r[j]
                        if c < max_r:
                            max_r = c
                    if max_r > best_r + 1e-8:
                        best_r = max_r
                        best_x, best_y = cx, cy
                x[k], y[k], r[k] = best_x, best_y, best_r

        # Final inflation pass
        for _ in range(10):
            order = rng.permutation(n_circles)
            improved = False
            for k in order:
                max_r = min(x[k], w - x[k], y[k], h - y[k])
                for j in range(n_circles):
                    if j == k:
                        continue
                    dx = x[k] - x[j]
                    dy = y[k] - y[j]
                    d = np.sqrt(dx * dx + dy * dy)
                    c = d - r[j]
                    if c < max_r:
                        max_r = c
                if max_r > r[k] + 1e-10:
                    r[k] = max_r - 1e-10
                    improved = True
            if not improved:
                break

        return x, y, r, w

    # --- SLSQP with explicit constraints ---
    def make_constraints():
        cons = []
        # Non-negative radii
        for i in range(N):
            cons.append({'type': 'ineq', 'fun': lambda z, i=i: z[2 * i + 2]})
        # Boundary: x_i >= r_i
        for i in range(N):
            cons.append({'type': 'ineq', 'fun': lambda z, i=i: z[2 * i] - z[2 * i + 2]})
        # Boundary: x_i + r_i <= w
        for i in range(N):
            cons.append({'type': 'ineq', 'fun': lambda z, i=i: z[-1] - z[2 * i] - z[2 * i + 2]})
        # Boundary: y_i >= r_i
        for i in range(N):
            cons.append({'type': 'ineq', 'fun': lambda z, i=i: z[2 * i + 1] - z[2 * i + 2]})
        # Boundary: y_i + r_i <= h = 2 - w
        for i in range(N):
            cons.append({'type': 'ineq', 'fun': lambda z, i=i: 2.0 - z[-1] - z[2 * i + 1] - z[2 * i + 2]})
        # Pairwise: d_ij >= r_i + r_j
        for i in range(N):
            for j in range(i + 1, N):
                cons.append({
                    'type': 'ineq',
                    'fun': lambda z, i=i, j=j: np.sqrt(
                        (z[2 * i] - z[2 * j]) ** 2 + (z[2 * i + 1] - z[2 * j + 1]) ** 2
                    ) - z[2 * i + 2] - z[2 * j + 2]
                })
        # w in [0.3, 1.7]
        cons.append({'type': 'ineq', 'fun': lambda z: z[-1] - 0.3})
        cons.append({'type': 'ineq', 'fun': lambda z: 1.7 - z[-1]})
        return cons

    def objective(z):
        r = z[2::3][:N]
        return -np.sum(r)

    def z_from_xywr(x, y, r, w):
        z = np.zeros(3 * N + 1)
        z[0::3][:N] = x
        z[1::3][:N] = y
        z[2::3][:N] = r
        z[-1] = w
        return z

    def xywr_from_z(z):
        x = z[0::3][:N].copy()
        y = z[1::3][:N].copy()
        r = z[2::3][:N].copy()
        w = z[-1]
        return x, y, r, w

    # --- Force-directed relaxation ---
    def force_relax(x, y, r, w, max_iter=300, dt=0.002):
        h = 2.0 - w
        x = x.copy()
        y = y.copy()
        for _ in range(max_iter):
            if time_left() < 5:
                break
            fx = np.zeros(N)
            fy = np.zeros(N)
            # Wall forces
            for i in range(N):
                if x[i] - r[i] < 0:
                    fx[i] += (x[i] - r[i]) * 30
                if x[i] + r[i] > w:
                    fx[i] -= (x[i] + r[i] - w) * 30
                if y[i] - r[i] < 0:
                    fy[i] += (y[i] - r[i]) * 30
                if y[i] + r[i] > h:
                    fy[i] -= (y[i] + r[i] - h) * 30
            # Pairwise
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx * dx + dy * dy)
                    min_d = r[i] + r[j]
                    if d < min_d and d > 1e-12:
                        ov = min_d - d
                        fx[i] += (dx / d) * ov * 15
                        fy[i] += (dy / d) * ov * 15
                        fx[j] -= (dx / d) * ov * 15
                        fy[j] -= (dy / d) * ov * 15
            x += dt * fx
            y += dt * fy
            for i in range(N):
                x[i] = np.clip(x[i], r[i], w - r[i])
                y[i] = np.clip(y[i], r[i], h - r[i])
        return x, y

    # --- Greedy radius growth ---
    def greedy_grow(x, y, r, w, max_rounds=20):
        h = 2.0 - w
        x = x.copy()
        y = y.copy()
        r = r.copy()
        rng = np.random.default_rng(99)
        for _ in range(max_rounds):
            if time_left() < 5:
                break
            order = rng.permutation(N)
            improved = False
            for k in order:
                max_r = min(x[k], w - x[k], y[k], h - y[k])
                for j in range(N):
                    if j == k:
                        continue
                    dx = x[k] - x[j]
                    dy = y[k] - y[j]
                    d = np.sqrt(dx * dx + dy * dy)
                    c = d - r[j]
                    if c < max_r:
                        max_r = c
                if max_r > r[k] + 1e-10:
                    r[k] = max_r - 1e-10
                    improved = True
            if not improved:
                break
        return x, y, r

    # --- Multi-start SLSQP polish ---
    def slsqp_polish(x, y, r, w, maxiter=200):
        z0 = z_from_xywr(x, y, r, w)
        bounds = [(0, 2)] * (3 * N) + [(0.3, 1.7)]
        cons = make_constraints()
        try:
            res = minimize(objective, z0, method='SLSQP', bounds=bounds,
                           constraints=cons,
                           options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 50})
            if res.success or res.fun < -np.sum(r):
                x2, y2, r2, w2 = xywr_from_z(res.x)
                return x2, y2, r2, w2
        except Exception:
            pass
        return x, y, r, w

    # --- Main optimization loop ---
    best_value = 0.0
    best = None

    # Aspect ratios to try
    aspect_ratios = [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.75, 0.55, 2.0, 0.5,
                     0.9, 1.1, 0.8, 1.2, 0.7, 1.4, 1.6, 0.55]

    # Phase 1: Pebble inflation for multiple aspect ratios
    for w0 in aspect_ratios:
        if time_left() < 15:
            break
        h0 = 2.0 - w0
        if h0 < 0.3:
            continue
        x, y, r, w = pebble_inflate(w0, h0, seed=int(w0 * 100))
        # Force relax to fix any overlaps
        x, y = force_relax(x, y, r, w, max_iter=200)
        # Greedy grow
        x, y, r = greedy_grow(x, y, r, w)
        valid, val = check_validity(x, y, r, w)
        if valid and val > best_value:
            best_value = val
            best = (x.copy(), y.copy(), r.copy(), w)

    # Phase 2: SLSQP polish on best results
    if best is not None and time_left() > 20:
        x, y, r, w = best
        x2, y2, r2, w2 = slsqp_polish(x, y, r, w, maxiter=300)
        valid, val = check_validity(x2, y2, r2, w2)
        if valid and val > best_value:
            best_value = val
            best = (x2.copy(), y2.copy(), r2.copy(), w2)

    # Phase 3: Try SLSQP from pebble results with different seeds
    for w0 in aspect_ratios[:8]:
        if time_left() < 10:
            break
        h0 = 2.0 - w0
        if h0 < 0.3:
            continue
        x, y, r, w = pebble_inflate(w0, h0, seed=int(w0 * 200) + 1)
        x, y = force_relax(x, y, r, w, max_iter=100)
        x, y, r = greedy_grow(x, y, r, w)
        x2, y2, r2, w2 = slsqp_polish(x, y, r, w, maxiter=150)
        valid, val = check_validity(x2, y2, r2, w2)
        if valid and val > best_value:
            best_value = val
            best = (x2.copy(), y2.copy(), r2.copy(), w2)

    # Phase 4: Final greedy growth on best
    if best is not None and time_left() > 5:
        x, y, r, w = best
        x, y, r = greedy_grow(x, y, r, w, max_rounds=30)
        valid, val = check_validity(x, y, r, w)
        if valid and val > best_value:
            best_value = val
            best = (x.copy(), y.copy(), r.copy(), w)

    # Phase 5: Final SLSQP polish
    if best is not None and time_left() > 10:
        x, y, r, w = best
        x2, y2, r2, w2 = slsqp_polish(x, y, r, w, maxiter=200)
        valid, val = check_validity(x2, y2, r2, w2)
        if valid and val > best_value:
            best_value = val
            best = (x2.copy(), y2.copy(), r2.copy(), w2)

    if best is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        w = 1.0
        h = 1.0
        x = np.array([(col + 0.5) * w / side for i in range(N) for col in range(side)][:N])
        y = np.array([(row + 0.5) * h / side for row in range(side) for i in range(N)][:N])
        r = np.full(N, min(w, h) / side * 0.45)
        return np.column_stack((x, y, r))

    x, y, r, w = best
    # Safety shrink
    r = r * (1 - 1e-7)
    return np.column_stack((x, y, r))
