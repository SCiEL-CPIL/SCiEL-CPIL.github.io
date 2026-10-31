"""Circle packing in perimeter-4 rectangle: pebble inflation + vectorized SLSQP polish + radius-growth."""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 21
START = time.time()
DEADLINE = START + 145.0  # safety margin before 150s kill


def _evaluate(x, W, H):
    """Return sum of radii if x is a valid packing, else None."""
    cx = x[:, 0]
    cy = x[:, 1]
    r = x[:, 2]
    n = len(cx)
    cx = np.clip(cx, 0.0, W)
    cy = np.clip(cy, 0.0, H)
    r = np.maximum(r, 1e-6)
    lo = np.minimum(np.minimum(cx, cy), np.minimum(W - cx, H - cy))
    r = np.minimum(r, lo)
    r = np.maximum(r, 1e-6)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(cx[i] - cx[j], cy[i] - cy[j])
            if d + 1e-7 < r[i] + r[j]:
                return None
    return float(np.sum(r))


def _pebble_inflate(W, H, n_circles, seed=42):
    """
    Sequential pebble inflation: add circles one at a time at the maximum radius
    allowed by the current free space, using a distance field.
    Returns array (n, 3) of [cx, cy, r].
    """
    rng = np.random.default_rng(seed)
    circles = []  # list of (cx, cy, r)
    
    # Use a grid to approximate the distance field
    grid_res = 200
    dx = W / grid_res
    dy = H / grid_res
    # Distance field: for each grid cell, the max radius a circle could have centered there
    dist_field = np.zeros((grid_res, grid_res))
    
    # Initialize with wall distances
    for iy in range(grid_res):
        for ix in range(grid_res):
            gx = (ix + 0.5) * dx
            gy = (iy + 0.5) * dy
            dist_field[iy, ix] = min(gx, W - gx, gy, H - gy)
    
    for c_idx in range(n_circles):
        if time.time() > DEADLINE - 10:
            break
        
        # Find the best location: maximize the local radius
        # Sample random points and also check the global max of the field
        best_pos = None
        best_r = 0
        
        # First try the global max of the current field
        max_idx = np.unravel_index(np.argmax(dist_field), dist_field.shape)
        gx = (max_idx[1] + 0.5) * dx
        gy = (max_idx[0] + 0.5) * dy
        candidate_r = dist_field[max_idx]
        if candidate_r > best_r:
            best_r = candidate_r
            best_pos = (gx, gy)
        
        # Also sample random points for diversity
        for _ in range(50):
            gx = rng.uniform(0, W)
            gy = rng.uniform(0, H)
            # Find nearest grid cell
            ix = min(int(gx / dx), grid_res - 1)
            iy = min(int(gy / dy), grid_res - 1)
            candidate_r = dist_field[iy, ix]
            if candidate_r > best_r:
                best_r = candidate_r
                best_pos = (gx, gy)
        
        if best_pos is None or best_r < 1e-4:
            # Fallback: place a tiny circle somewhere
            gx = rng.uniform(0.01 * W, 0.99 * W)
            gy = rng.uniform(0.01 * H, 0.99 * H)
            best_r = min(gx, W - gx, gy, H - gy) * 0.3
            best_pos = (gx, gy)
        
        # Verify against existing circles
        cx, cy = best_pos
        r = best_r
        # Check against all existing circles
        for (ocx, ocy, or_) in circles:
            d = np.hypot(cx - ocx, cy - ocy)
            max_r_allowed = d - or_
            if max_r_allowed < r:
                r = max_r_allowed
        # Check walls
        r = min(r, cx, W - cx, cy, H - cy)
        r = max(r, 1e-4)
        
        circles.append((cx, cy, r))
        
        # Update distance field: subtract this circle's influence
        # For each grid cell, the new max radius is min(old, distance_to_circle - r)
        for iy in range(grid_res):
            for ix in range(grid_res):
                gx = (ix + 0.5) * dx
                gy = (iy + 0.5) * dy
                d = np.hypot(gx - cx, gy - cy)
                new_r = d - r
                if new_r < dist_field[iy, ix]:
                    dist_field[iy, ix] = new_r
    
    arr = np.array(circles)
    if len(arr) < N:
        while len(arr) < N:
            arr = np.vstack([arr, [0.01, 0.01, 0.005]])
    return arr[:N]


def _force_relax(z, max_iter=150):
    """Force-directed relaxation in unified parameter space to resolve overlaps."""
    z = z.copy().astype(float)
    n = N
    dt = 0.003

    for _ in range(max_iter):
        if time.time() > DEADLINE - 5:
            break
        x = z[0::3][:n]
        y = z[1::3][:n]
        r = z[2::3][:n]
        w = z[-1]
        h = 2.0 - w

        fx = np.zeros(n)
        fy = np.zeros(n)

        # Wall repulsion
        for i in range(n):
            if x[i] - r[i] < 0:
                fx[i] += (x[i] - r[i]) * 20
            if x[i] + r[i] > w:
                fx[i] -= (x[i] + r[i] - w) * 20
            if y[i] - r[i] < 0:
                fy[i] += (y[i] - r[i]) * 20
            if y[i] + r[i] > h:
                fy[i] -= (y[i] + r[i] - h) * 20

        # Pairwise repulsion
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                min_d = r[i] + r[j]
                if d < min_d and d > 1e-12:
                    overlap = min_d - d
                    fx[i] += (dx / d) * overlap * 10
                    fy[i] += (dy / d) * overlap * 10
                    fx[j] -= (dx / d) * overlap * 10
                    fy[j] -= (dy / d) * overlap * 10

        # Update positions
        z[0::3][:n] += dt * fx
        z[1::3][:n] += dt * fy

        # Clamp positions
        for i in range(n):
            z[3 * i] = np.clip(z[3 * i], r[i], w - r[i])
            z[3 * i + 1] = np.clip(z[3 * i + 1], r[i], h - r[i])

    return z


def _polish_result(z, time_budget):
    """
    Polish a valid packing by iteratively growing each radius to its local slack
    and re-relaxing positions.
    """
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    start = time.time()
    max_outer = 30

    for outer in range(max_outer):
        if time.time() - start > time_budget:
            break

        # Try to grow each radius to its local slack
        grew = False
        for i in range(n):
            # Boundary slack
            slack = min(x[i] - r[i], w - x[i] - r[i], y[i] - r[i], h - y[i] - r[i])

            # Pairwise slack
            for j in range(n):
                if i == i:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                slack = min(slack, d - r[i] - r[j])

            if slack > 1e-7:
                growth = min(0.002, slack * 0.5)
                r[i] += growth
                grew = True

        # After growing, do a light position relaxation to resolve any new overlaps
        for _ in range(20):
            if time.time() - start > time_budget:
                break
            fx = np.zeros(n)
            fy = np.zeros(n)

            for i in range(n):
                if x[i] - r[i] < 0:
                    fx[i] += (x[i] - r[i]) * 20
                if x[i] + r[i] > w:
                    fx[i] -= (x[i] + r[i] - w) * 20
                if y[i] - r[i] < 0:
                    fy[i] += (y[i] - r[i]) * 20
                if y[i] + r[i] > h:
                    fy[i] -= (y[i] + r[i] - h) * 20

            for i in range(n):
                for j in range(i + 1, n):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx * dx + dy * dy)
                    min_d = r[i] + r[j]
                    if d < min_d and d > 1e-12:
                        overlap = min_d - d
                        fx[i] += (dx / d) * overlap * 10
                        fy[i] += (dy / d) * overlap * 10
                        fx[j] -= (dx / d) * overlap * 10
                        fy[j] -= (dy / d) * overlap * 10

            dt = 0.001
            x += dt * fx
            y += dt * fy

            for i in range(n):
                x[i] = np.clip(x[i], r[i], w - r[i])
                y[i] = np.clip(y[i], r[i], h - r[i])

        if not grew:
            break

    z[0::3][:n] = x
    z[1::3][:n] = y
    z[2::3][:n] = r

    return z


def _slsqp_vectorized(z0, max_iter=300):
    """
    SLSQP optimization with vectorized constraints for speed.
    z: flat vector [x1, y1, r1, ..., xN, yN, rN, w]
    Returns (z_opt, value) or (None, 0.0).
    """
    n = N
    dim = 3 * N + 1

    def objective(z):
        r = z[2::3][:n]
        w = z[-1]
        h = 2.0 - w
        if w < 0.3 or h < 0.3:
            return 1e6
        return -np.sum(r)

    def cons_inside(z):
        x = z[0::3][:n]
        y = z[1::3][:n]
        r = z[2::3][:n]
        w = z[-1]
        h = 2.0 - w
        return np.concatenate([
            x - r,
            w - x - r,
            y - r,
            h - y - r
        ])

    def cons_non_overlap(z):
        x = z[0::3][:n]
        y = z[1::3][:n]
        r = z[2::3][:n]
        cons = []
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy + 1e-12)
                cons.append(dist - r[i] - r[j])
        return np.array(cons)

    def cons_positive_r(z):
        r = z[2::3][:n]
        return r

    def cons_aspect(z):
        w = z[-1]
        h = 2.0 - w
        return np.array([w - 0.4, 1.6 - w])

    constraints = [
        {'type': 'ineq', 'fun': cons_inside},
        {'type': 'ineq', 'fun': cons_non_overlap},
        {'type': 'ineq', 'fun': cons_positive_r},
        {'type': 'ineq', 'fun': cons_aspect}
    ]

    def check_validity(z, tol=1e-7):
        x = z[0::3][:n]
        y = z[1::3][:n]
        r = z[2::3][:n]
        w = z[-1]
        h = 2.0 - w

        if w <= 0.05 or h <= 0.05:
            return False, 0.0

        for i in range(n):
            if r[i] <= 0:
                return False, 0.0
            if x[i] < r[i] - tol or x[i] > w - r[i] + tol:
                return False, 0.0
            if y[i] < r[i] - tol or y[i] > h - r[i] + tol:
                return False, 0.0

        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d2 = dx * dx + dy * dy
                r_sum = r[i] + r[j]
                if d2 < r_sum ** 2 - tol:
                    return False, 0.0

        val = float(np.sum(r))
        return True, val

    try:
        result = minimize(
            objective, z0, method='SLSQP',
            bounds=[(0, 2)] * (3 * n) + [(0.3, 1.7)],
            constraints=constraints,
            options={'maxiter': max_iter, 'ftol': 1e-10, 'maxls': 50}
        )

        z_opt = result.x
        valid, val = check_validity(z_opt)

        if valid and val > 0:
            return z_opt, val
        else:
            # Try repair: shrink radii slightly
            z_repair = z_opt.copy()
            r_idx = np.arange(2, len(z_repair) - 1, 3)
            z_repair[r_idx] *= 0.95
            valid, val = check_validity(z_repair)
            if valid and val > 0:
                return z_repair, val
    except Exception:
        pass

    return None, 0.0


def _grid_layout(kx, ky, W, H, shrink=0.95):
    """Build a kx*ky grid layout (pad/truncate to N)."""
    cx = np.linspace(0.0, W, kx + 2)[1:-1]
    cy = np.linspace(0.0, H, ky + 2)[1:-1]
    pts = []
    for yy in cy:
        for xx in cx:
            r = min(xx, yy, W - xx, H - yy) * shrink
            pts.append([xx, yy, max(r, 1e-4)])
    arr = np.array(pts)
    if len(arr) < N:
        while len(arr) < N:
            arr = np.vstack([arr, [0.01, 0.01, 0.005]])
    return arr[:N]


def _honeycomb_layout(W, H):
    """Generate a hexagonal/honeycomb-like layout for better packing density."""
    pts = []
    base_r = min(W, H) / 8.0

    rows = int(H / (base_r * 1.732)) + 2
    cols = int(W / (base_r * 1.5)) + 2

    for row in range(rows):
        y = row * base_r * 1.732
        if y > H:
            break
        offset = (base_r * 1.5) if (row % 2 == 1) else 0
        for col in range(cols):
            x = col * base_r * 1.5 + offset
            if x > W:
                break
            r = min(x, y, W - x, H - y) * 0.9
            r = max(r, 1e-4)
            pts.append([x, y, r])

    arr = np.array(pts)
    if len(arr) < N:
        while len(arr) < N:
            arr = np.vstack([arr, [0.01, 0.01, 0.005]])
    return arr[:N]


def construct():
    best_sum = 0.0
    best_z = None  # unified parameter vector

    def time_ok():
        return time.time() < DEADLINE - 3

    random.seed(42)
    np.random.seed(42)

    candidates = []

    # --- Phase 0: Pebble inflation for diverse aspect ratios ---
    Ws_pebble = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
    for W in Ws_pebble:
        H = 2.0 - W
        if H < 0.3 or W < 0.3:
            continue
        for seed in [42, 123, 456]:
            if not time_ok():
                break
            g = _pebble_inflate(W, H, N, seed=seed)
            s = _evaluate(g, W, H)
            if s is not None:
                z = np.zeros(3 * N + 1)
                z[0::3][:N] = g[:, 0]
                z[1::3][:N] = g[:, 1]
                z[2::3][:N] = g[:, 2]
                z[-1] = W
                candidates.append((s, z))

    # --- Phase 0b: Grid and honeycomb layouts ---
    grid_sizes = [(3, 4), (4, 3), (3, 3), (4, 4), (5, 3), (3, 5)]
    for W in Ws_pebble:
        H = 2.0 - W
        if H < 0.3 or W < 0.3:
            continue
        for kx, ky in grid_sizes:
            if kx * ky < N:
                continue
            g = _grid_layout(kx, ky, W, H, shrink=0.95)
            s = _evaluate(g, W, H)
            if s is not None:
                z = np.zeros(3 * N + 1)
                z[0::3][:N] = g[:, 0]
                z[1::3][:N] = g[:, 1]
                z[2::3][:N] = g[:, 2]
                z[-1] = W
                candidates.append((s, z))

        h = _honeycomb_layout(W, H)
        s_h = _evaluate(h, W, H)
        if s_h is not None:
            z = np.zeros(3 * N + 1)
            z[0::3][:N] = h[:, 0]
            z[1::3][:N] = h[:, 1]
            z[2::3][:N] = h[:, 2]
            z[-1] = W
            candidates.append((s_h, z))

    # Sort by initial score
    candidates.sort(key=lambda t: -t[0])

    # --- Phase 1: Force relaxation + SLSQP on top candidates ---
    top_k = min(12, len(candidates))
    for idx in range(top_k):
        if not time_ok():
            break
        s0, z0 = candidates[idx]

        # Force relaxation
        z_relaxed = _force_relax(z0, max_iter=150)

        # SLSQP optimization
        z_opt, s_opt = _slsqp_vectorized(z_relaxed, max_iter=250)
        if z_opt is not None and s_opt > best_sum:
            best_sum = s_opt
            best_z = z_opt

    # --- Phase 2: Polish the best result ---
    if time_ok() and best_z is not None:
        remaining = DEADLINE - time.time()
        polished = _polish_result(best_z, time_budget=min(30, remaining - 5))
        if polished is not None:
            x = polished[0::3][:N]
            y = polished[1::3][:N]
            r = polished[2::3][:N]
            w = polished[-1]
            h = 2.0 - w
            x_arr = np.column_stack([x, y, r])
            s_polished = _evaluate(x_arr, w, h)
            if s_polished is not None and s_polished > best_sum:
                best_sum = s_polished
                best_z = polished

    # Fallback
    if best_z is None:
        W, H = 1.0, 1.0
        g = _grid_layout(5, 5, W, H, shrink=0.95)
        best_z = np.zeros(3 * N + 1)
        best_z[0::3][:N] = g[:, 0]
        best_z[1::3][:N] = g[:, 1]
        best_z[2::3][:N] = g[:, 2]
        best_z[-1] = W

    # Extract final result
    cx = best_z[0::3][:N]
    cy = best_z[1::3][:N]
    r = best_z[2::3][:N]
    W = best_z[-1]
    H = 2.0 - W

    # Final clamping
    cx = np.clip(cx, 0.0, W)
    cy = np.clip(cy, 0.0, H)
    r = np.maximum(r, 1e-6)
    lo = np.minimum(np.minimum(cx, cy), np.minimum(W - cx, H - cy))
    r = np.minimum(r, lo)
    r = np.maximum(r, 1e-6)
    out = np.column_stack([cx, cy, r])

    # Final validation
    s_final = _evaluate(out, W, H)
    if s_final is None:
        side = 5
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-4)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    return out
