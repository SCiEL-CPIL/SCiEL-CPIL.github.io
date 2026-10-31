"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _pack_rect(cx, cy, r, W, H):
    """Check if circles fit in rectangle [0,W]x[0,H] with disjoint interiors."""
    if W <= 0 or H <= 0:
        return False
    if np.any(cx - r < -1e-9) or np.any(cy - r < -1e-9):
        return False
    if np.any(cx + r > W + 1e-9) or np.any(cy + r > H + 1e-9):
        return False
    n = len(cx)
    for i in range(n):
        for j in range(i + 1, n):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            if dx * dx + dy * dy < (r[i] + r[j]) ** 2 - 1e-9:
                return False
    return True


def _radius_sum(cx, cy, r, W, H):
    if _pack_rect(cx, cy, r, W, H):
        return float(np.sum(r))
    return float('-inf')


def _force_directed_relax(cx, cy, r, W, H, max_iter=3000, dt=0.0005):
    """Force-directed relaxation to fix overlaps and wall violations."""
    n = len(cx)
    cx = cx.copy()
    cy = cy.copy()
    for _ in range(max_iter):
        fx = np.zeros(n)
        fy = np.zeros(n)
        for i in range(n):
            if cx[i] - r[i] < 0:
                fx[i] += (cx[i] - r[i]) * 10
            if cy[i] - r[i] < 0:
                fy[i] += (cy[i] - r[i]) * 10
            if cx[i] + r[i] > W:
                fx[i] -= (cx[i] + r[i] - W) * 10
            if cy[i] + r[i] > H:
                fy[i] -= (cy[i] + r[i] - H) * 10
        for i in range(n):
            for j in range(i + 1, n):
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                d = np.sqrt(dx * dx + dy * dy)
                min_d = r[i] + r[j]
                if d < min_d and d > 1e-12:
                    overlap = min_d - d
                    fx[i] += (dx / d) * overlap * 5
                    fy[i] += (dy / d) * overlap * 5
                    fx[j] -= (dx / d) * overlap * 5
                    fy[j] -= (dy / d) * overlap * 5
        cx += dt * fx
        cy += dt * fy
        cx = np.clip(cx, r, W - r)
        cy = np.clip(cy, r, H - r)
    return cx, cy


def _hex_init(W, H, seed):
    """Hexagonal (staggered) grid initialization that fills the rectangle."""
    rng = np.random.default_rng(seed)
    # Choose a reasonable cell size so we get ~N circles
    # Hexagonal packing: horizontal spacing = 2r, vertical = sqrt(3) r
    # We want ~N circles. Use a grid of cols x rows.
    cols = int(np.round(np.sqrt(N * W / H)))
    cols = max(2, min(cols, 12))
    rows = int(np.ceil(N / cols))
    # Fit hexagonal lattice into rectangle
    # horizontal spacing
    dx = W / cols
    dy = H / rows
    r0 = min(dx, dy) * 0.48
    cx = np.zeros(N)
    cy = np.zeros(N)
    r = np.zeros(N)
    count = 0
    for j in range(rows):
        for i in range(cols):
            if count >= N:
                break
            # stagger odd rows
            x = (i + 0.5) * dx + (0.5 * dx if (j % 2) else 0.0)
            y = (j + 0.5) * dy
            # small jitter to break symmetry
            x += rng.uniform(-0.05 * dx, 0.05 * dx)
            y += rng.uniform(-0.05 * dy, 0.05 * dy)
            cx[count] = x
            cy[count] = y
            r[count] = r0
            count += 1
    return cx, cy, r


def _random_init(W, H, seed):
    """Random positions, uniform small radii."""
    rng = np.random.default_rng(seed)
    cx = rng.uniform(0, W, N)
    cy = rng.uniform(0, H, N)
    r = np.full(N, min(W, H) * 0.12)
    return cx, cy, r


def _slsqp_polish(cx, cy, r, W, H, max_iter=400):
    """Light SLSQP polish to grow radii while keeping feasibility."""
    def objective(x):
        cx = x[:N]
        cy = x[N:2*N]
        r = x[2*N:3*N]
        pen = 0.0
        for i in range(N):
            if cx[i] - r[i] < 0:
                pen += (cx[i] - r[i]) ** 2
            if cy[i] - r[i] < 0:
                pen += (cy[i] - r[i]) ** 2
            if cx[i] + r[i] > W:
                pen += (cx[i] + r[i] - W) ** 2
            if cy[i] + r[i] > H:
                pen += (cy[i] + r[i] - H) ** 2
        for i in range(N):
            for j in range(i + 1, N):
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                d = np.sqrt(dx * dx + dy * dy)
                if d < r[i] + r[j]:
                    pen += (r[i] + r[j] - d) ** 2
        return -np.sum(r) + 10 * pen

    x0 = np.concatenate([cx, cy, r])
    bounds = []
    for _ in range(N):
        bounds.append((0, W))
    for _ in range(N):
        bounds.append((0, H))
    for _ in range(N):
        bounds.append((1e-4, min(W, H) / 2))
    try:
        res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                       options={'maxiter': max_iter, 'ftol': 1e-9})
        cx_opt = res.x[:N]
        cy_opt = res.x[N:2*N]
        r_opt = res.x[2*N:3*N]
        if _pack_rect(cx_opt, cy_opt, r_opt, W, H):
            return cx_opt, cy_opt, r_opt
    except Exception:
        pass
    return None, None, None


def _grow_radii(cx, cy, r, W, H, steps=60):
    """Iteratively grow all radii and re-relax; keep if still feasible."""
    for _ in range(steps):
        r_new = r * 1.01
        cx2, cy2 = _force_directed_relax(cx, cy, r_new, W, H, max_iter=800, dt=0.0004)
        if _pack_rect(cx2, cy2, r_new, W, H):
            cx, cy, r = cx2, cy2, r_new
        else:
            break
    return cx, cy, r


def _optimize_for_aspect(W, H, deadline, seed_base):
    """Random-restart search over hexagonal/random starts for a fixed aspect ratio."""
    best_val = -float('inf')
    best_result = None
    attempt = 0
    while time.time() < deadline:
        seed = seed_base + attempt
        if attempt % 2 == 0:
            cx, cy, r = _hex_init(W, H, seed)
        else:
            cx, cy, r = _random_init(W, H, seed)
        # Relax to fix initial overlaps
        cx, cy = _force_directed_relax(cx, cy, r, W, H, max_iter=1500, dt=0.0005)
        # Grow radii
        cx, cy, r = _grow_radii(cx, cy, r, W, H, steps=40)
        val = _radius_sum(cx, cy, r, W, H)
        if val > best_val:
            best_val = val
            best_result = (cx.copy(), cy.copy(), r.copy(), W, H)
        # SLSQP polish on the best-so-far
        if time.time() < deadline - 2.0:
            cxp, cyp, rp = _slsqp_polish(cx, cy, r, W, H, max_iter=300)
            if cxp is not None:
                cxp, cyp, rp = _grow_radii(cxp, cyp, rp, W, H, steps=30)
                valp = _radius_sum(cxp, cyp, rp, W, H)
                if valp > best_val:
                    best_val = valp
                    best_result = (cxp.copy(), cyp.copy(), rp.copy(), W, H)
        attempt += 1
    return best_result, best_val


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    best_val = -float('inf')
    best_result = None

    # Try various aspect ratios (W + H = 2, so H = 2 - W)
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.5, 1.1, 0.9]

    for W in aspect_ratios:
        if time.time() > deadline:
            break
        H = 2.0 - W
        if H <= 0:
            continue
        result, val = _optimize_for_aspect(W, H, deadline, seed_base=int(W * 1000))
        if val > best_val:
            best_val = val
            best_result = result

    # Fallback to seed grid
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    cx, cy, r, W, H = best_result
    return np.column_stack([cx, cy, r])
# EVOLVE-BLOCK-END
