"""Circle packing in perimeter-4 rectangle: force-directed pressure simulation."""
import numpy as np
import time

N = 21


def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    best_value = 0.0
    best_result = None

    def check_validity(x, y, r, w, h, tol=1e-9):
        if w <= 0.05 or h <= 0.05:
            return False, 0.0
        if np.any(r <= 0):
            return False, 0.0
        if np.any(x < r - tol) or np.any(x > w - r + tol):
            return False, 0.0
        if np.any(y < r - tol) or np.any(y > h - r + tol):
            return False, 0.0
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d2 = dx * dx + dy * dy
        rs = r[:, None] + r[None, :]
        iu = np.triu_indices(N, k=1)
        if np.any(d2[iu] < rs[iu] ** 2 - tol):
            return False, 0.0
        return True, float(np.sum(r))

    def simulate(w0, h0, time_budget, seed=0):
        """Run force-directed pressure simulation for a given rectangle."""
        rng = np.random.default_rng(seed)
        x = np.empty(N)
        y = np.empty(N)
        r = np.empty(N)
        for i in range(N):
            x[i] = rng.uniform(0.1, w0 - 0.1)
            y[i] = rng.uniform(0.1, h0 - 0.1)
            r[i] = rng.uniform(0.04, 0.09)

        # Initial relaxation to resolve gross overlaps (no inflation yet)
        vx = np.zeros(N)
        vy = np.zeros(N)
        dt = 0.004
        damping = 0.88
        iu = np.triu_indices(N, k=1)

        def forces(x, y, r, inflate):
            fx = np.zeros(N)
            fy = np.zeros(N)
            # Wall repulsion
            fx -= np.maximum(r - x, 0.0) * 60.0
            fx += np.maximum(x + r - w0, 0.0) * 60.0
            fy -= np.maximum(r - y, 0.0) * 60.0
            fy += np.maximum(y + r - h0, 0.0) * 60.0
            # Center attraction
            fx += (w0 * 0.5 - x) * 0.5
            fy += (h0 * 0.5 - y) * 0.5
            # Pairwise repulsion
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            d2 = dx * dx + dy * dy
            rs = r[:, None] + r[None, :]
            dp = np.sqrt(np.maximum(d2[iu], 1e-12))
            overlap = np.maximum(rs[iu] - dp, 0.0)
            fp = overlap * 22.0
            ux = dx[iu] / dp
            uy = dy[iu] / dp
            for k in range(len(iu[0])):
                a = iu[0][k]; b = iu[1][k]
                fx[a] += ux[k] * fp[k]; fx[b] -= ux[k] * fp[k]
                fy[a] += uy[k] * fp[k]; fy[b] -= uy[k] * fp[k]
            return fx, fy

        def step(x, y, r, vx, vy, dt, damping, inflate):
            fx, fy = forces(x, y, r, inflate)
            vx = vx * damping + fx * dt
            vy = vy * damping + fy * dt
            x = x + vx * dt
            y = y + vy * dt
            x = np.clip(x, r, w0 - r)
            y = np.clip(y, r, h0 - r)
            return x, y, vx, vy

        # Phase A: settle positions (inflate=False)
        for it in range(600):
            x, y, vx, vy = step(x, y, r, vx, vy, dt, damping, False)
            if time.time() > start_time + time_budget:
                break

        # Phase B: inflate radii gradually
        base = r.copy()
        max_growth = 0.0
        for phase in range(30):
            target = 1.0 + 0.05 * phase
            for sub in range(100):
                if time.time() > start_time + time_budget:
                    break
                r = base * (1.0 + (target - 1.0) * (sub + 1) / 100.0)
                x, y, vx, vy = step(x, y, r, vx, vy, dt, damping, True)
                if sub == 99:
                    cur = float(np.sum(r))
                    if cur > max_growth:
                        max_growth = cur
        return x, y, r, w0, h0

    def optimize_radii(x, y, r, w, h, time_budget):
        """Iteratively maximize each radius, then re-optimize positions slightly."""
        r = r.copy()
        iu = np.triu_indices(N, k=1)
        for _ in range(5):
            if time.time() > start_time + time_budget:
                break
            for i in range(N):
                # Calculate max possible radius for circle i given current positions of others
                r_max = min(x[i], w - x[i], y[i], h - y[i])
                for j in range(N):
                    if i == j:
                        continue
                    d = np.hypot(x[i] - x[j], y[i] - y[j])
                    r_max = min(r_max, d - r[j])
                r_max = max(0.0, r_max)
                if r_max > r[i]:
                    r[i] = r_max
            # After increasing radii, we might need to relax positions to allow further growth
            # Do a few steps of force simulation with the new radii to resolve overlaps if any (shouldn't be any if valid)
            # But if we just set r[i] = d - r[j], it's touching. To allow *more* growth, we need to move things.
            # So we run a few steps of simulation to spread them out.
            vx = np.zeros(N)
            vy = np.zeros(N)
            dt = 0.002
            damping = 0.9
            for _ in range(50):
                if time.time() > start_time + time_budget:
                    break
                fx, fy = forces(x, y, r, True)
                vx = vx * damping + fx * dt
                vy = vy * damping + fy * dt
                x = x + vx * dt
                y = y + vy * dt
                x = np.clip(x, r, w - r)
                y = np.clip(y, r, h - r)
        return x, y, r

    def relocate_polish(x, y, r, w, h, time_budget):
        """Nudge each circle to its max-radius position, then re-expand."""
        x = x.copy(); y = y.copy(); r = r.copy()
        for rnd in range(12):
            if time.time() > start_time + time_budget:
                break
            improved = False
            for i in range(N):
                # max radius at current position
                r_wall = min(x[i], w - x[i], y[i], h - y[i])
                rmax = r_wall
                for j in range(N):
                    if i == j:
                        continue
                    d = np.hypot(x[i] - x[j], y[i] - y[j])
                    rmax = min(rmax, d - r[j])
                rmax = max(0.0, rmax)
                if rmax > r[i] + 1e-6:
                    r[i] = rmax
                    improved = True
                # try small moves to open more space
                # We want to move circle i to increase the min clearance to neighbors/walls
                # Simple heuristic: try moving towards the "center of free space" or just random small steps
                # Here we try a set of small steps
                for dx, dy in [(0.01, 0), (-0.01, 0), (0, 0.01), (0, -0.01),
                               (0.007, 0.007), (-0.007, 0.007), (0.007, -0.007), (-0.007, -0.007)]:
                    nx = x[i] + dx
                    ny = y[i] + dy
                    if nx < r[i] or nx > w - r[i] or ny < r[i] or ny > h - r[i]:
                        continue
                    ok = True
                    for j in range(N):
                        if i == j:
                            continue
                        d2 = (nx - x[j]) ** 2 + (ny - y[j]) ** 2
                        if d2 < (r[i] + r[j]) ** 2 - 1e-9:
                            ok = False
                            break
                    if ok:
                        # Check if this move allows for a larger radius
                        r_wall_new = min(nx, w - nx, ny, h - ny)
                        rmax_new = r_wall_new
                        for j in range(N):
                            if i == j:
                                continue
                            d = np.hypot(nx - x[j], ny - y[j])
                            rmax_new = min(rmax_new, d - r[j])
                        rmax_new = max(0.0, rmax_new)
                        if rmax_new > r[i] + 1e-6:
                            x[i] = nx
                            y[i] = ny
                            r[i] = rmax_new
                            improved = True
                            break
            if not improved:
                break
        return x, y, r

    # Sweep a range of aspect ratios
    # Include geometric ratios and finer sweep
    aspect_ratios = [
        1.0, 0.9, 1.1, 0.8, 1.2, 0.7, 1.3, 0.6, 1.4, 0.5, 1.5,
        np.sqrt(2), 1/np.sqrt(2), np.sqrt(3), 1/np.sqrt(3),
        np.pi/2, 2/np.pi, 1.414, 0.707, 1.732, 0.577,
        1.618, 0.618 # Golden ratio
    ]
    # Sort and remove duplicates
    aspect_ratios = sorted(list(set(np.round(aspect_ratios, 3))))
    
    for k, w0 in enumerate(aspect_ratios):
        h0 = 2.0 - w0
        if h0 < 0.35 or w0 < 0.35:
            continue
        remaining = deadline - (time.time() - start_time)
        if remaining < 3.0:
            break
        budget = min(6.0, remaining - 1.0)
        try:
            x, y, r, w, h = simulate(w0, h0, budget, seed=1000 + k)
            # Optimize radii (iterative growth + relaxation)
            x, y, r = optimize_radii(x, y, r, w, h, deadline - (time.time() - start_time) - 2)
            # Relocate polish
            x, y, r = relocate_polish(x, y, r, w, h, deadline - (time.time() - start_time) - 2)
            # Final optimize radii pass
            x, y, r = optimize_radii(x, y, r, w, h, deadline - (time.time() - start_time) - 1)
            valid, val = check_validity(x, y, r, w, h)
            if valid and val > best_value:
                best_value = val
                best_result = (x.copy(), y.copy(), r.copy(), w, h)
        except Exception:
            continue

    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    x, y, r, w, h = best_result
    # Final safety shrink
    for _ in range(8):
        rt = r * (1 - 1e-9)
        if check_validity(x, y, rt, w, h)[0]:
            r = rt
        else:
            break

    return np.column_stack((x, y, r))
