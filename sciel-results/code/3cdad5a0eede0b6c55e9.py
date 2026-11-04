"""Circle packing in unit square: pack 26 circles, maximize sum of radii."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _eval_packing(xy, r):
    """
    Evaluate a packing: returns (total_radius, is_valid).
    Valid means all circles inside [0,1]^2 and disjoint interiors.
    """
    n = len(xy)
    for i in range(n):
        x, y, ri = xy[i, 0], xy[i, 1], r[i]
        if ri < -1e-9:
            return 0.0, False
        if x - ri < -1e-9 or 1.0 - x - ri < -1e-9 or y - ri < -1e-9 or 1.0 - y - ri < -1e-9:
            return 0.0, False
    for i in range(n):
        for j in range(i + 1, n):
            dx = xy[i, 0] - xy[j, 0]
            dy = xy[i, 1] - xy[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - 1e-9:
                return 0.0, False
    return float(np.sum(r)), True


def _hex_init(n, seed=0, jitter=0.0, scale=0.78):
    """Hexagonal lattice initialization covering the unit square."""
    rng = np.random.RandomState(seed)
    # Hex lattice spacing
    a = scale / np.sqrt(3)  # horizontal spacing
    dx = a
    dy = a * np.sqrt(3)
    pos = []
    row = 0
    while len(pos) < n:
        offset = (dx / 2) if (row % 2) else 0.0
        col = 0
        while len(pos) < n:
            x = 0.5 * dx + col * dx + offset
            y = 0.5 * dy + row * dy
            if x < 1.0 and y < 1.0:
                pos.append([x, y])
            col += 1
        row += 1
    pos = np.array(pos[:n])
    if jitter > 0:
        pos = pos + rng.uniform(-jitter, jitter, size=pos.shape)
    pos = np.clip(pos, 0.05, 0.95)
    # Initial radii: half the minimum spacing, capped
    r = np.full(n, dx * 0.48)
    # Clip to walls
    for i in range(n):
        r[i] = min(r[i], pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
        r[i] = max(r[i], 0.01)
    return pos, r


def _random_init(n, seed=0):
    """Random initialization with moderate radii."""
    rng = np.random.RandomState(seed)
    xy = rng.uniform(0.08, 0.92, size=(n, 2))
    r = rng.uniform(0.03, 0.07, size=n)
    return xy, r


def _smooth_objective(v, n, beta, lam):
    """
    Smooth objective: minimize -sum(r) + lam * (wall_pen + pair_pen).
    Uses softplus for violations.
    """
    x = v[0:n]
    y = v[n:2*n]
    r = v[2*n:3*n]

    wall_min = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    wall_viol = r - wall_min
    wall_pen = np.sum(np.logaddexp(0.0, beta * wall_viol) / beta)

    total_pair_pen = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            viol = r[i] + r[j] - dist
            total_pair_pen += np.logaddexp(0.0, beta * viol) / beta

    return -np.sum(r) + lam * (wall_pen + total_pair_pen)


def _smooth_grad(v, n, beta, lam):
    """
    Gradient of the smooth objective.
    """
    x = v[0:n].copy()
    y = v[n:2*n].copy()
    r = v[2*n:3*n].copy()

    grad = np.zeros(3 * n)

    # Wall terms
    wall_min = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    wall_viol = r - wall_min
    sig_wall = 1.0 / (1.0 + np.exp(-beta * wall_viol))

    for i in range(n):
        s = sig_wall[i]
        if s < 1e-12:
            continue
        dists = np.array([x[i], 1.0 - x[i], y[i], 1.0 - y[i]])
        min_idx = int(np.argmin(dists))
        f = lam * s
        if min_idx == 0:
            grad[2*n + i] += f
            grad[i] -= f
        elif min_idx == 1:
            grad[2*n + i] += f
            grad[i] += f
        elif min_idx == 2:
            grad[2*n + i] += f
            grad[n + i] -= f
        else:
            grad[2*n + i] += f
            grad[n + i] += f

    # Pairwise terms
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist_sq = dx * dx + dy * dy
            if dist_sq < 1e-20:
                continue
            dist = np.sqrt(dist_sq)
            viol = r[i] + r[j] - dist
            s = 1.0 / (1.0 + np.exp(-beta * viol))
            if s < 1e-12:
                continue
            f = lam * s
            grad[i] += f * dx / dist
            grad[n + i] += f * dy / dist
            grad[j] -= f * dx / dist
            grad[n + j] -= f * dy / dist
            grad[2*n + i] += f
            grad[2*n + j] += f

    # Objective: -sum(r)
    grad[2*n:3*n] -= 1.0

    return grad


def _lbfgs_step(v0, n, beta, lam, deadline, max_iter=2000, maxfun=10000):
    """Run one L-BFGS-B step with given beta, lam."""
    bounds = []
    for i in range(n):
        bounds.append((0.001, 0.999))
    for i in range(n):
        bounds.append((0.001, 0.999))
    for i in range(n):
        bounds.append((0.001, 0.4999))

    def obj(v):
        if time.time() > deadline:
            return float('inf')
        return _smooth_objective(v, n, beta, lam)

    def grad(v):
        if time.time() > deadline:
            return np.zeros(3 * n)
        return _smooth_grad(v, n, beta, lam)

    try:
        result = minimize(obj, v0, jac=grad, method='L-BFGS-B',
                          bounds=bounds,
                          options={'maxiter': max_iter, 'ftol': 1e-10,
                                   'gtol': 1e-7, 'maxfun': maxfun})
        return result.x
    except Exception:
        return v0


def _feasibility_projection(xy, r, iters=50):
    """
    Project to feasibility: iteratively shrink radii and push circles apart.
    """
    n = len(xy)
    xy = xy.copy()
    r = r.copy()
    for it in range(iters):
        # Push circles apart
        for i in range(n):
            for j in range(i + 1, n):
                dx = xy[j, 0] - xy[i, 0]
                dy = xy[j, 1] - xy[i, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = r[i] + r[j]
                if dist < min_dist:
                    if dist < 1e-12:
                        dx, dy = 1.0, 0.0
                        dist = 1.0
                    overlap = min_dist - dist
                    push = overlap / 2.0 + 1e-6
                    ux, uy = dx / dist, dy / dist
                    xy[i, 0] -= push * ux
                    xy[i, 1] -= push * uy
                    xy[j, 0] += push * ux
                    xy[j, 1] += push * uy
        # Clip to walls and shrink radii to fit
        for i in range(n):
            xy[i, 0] = np.clip(xy[i, 0], 0.0, 1.0)
            xy[i, 1] = np.clip(xy[i, 1], 0.0, 1.0)
            wall_min = min(xy[i, 0], 1.0 - xy[i, 0], xy[i, 1], 1.0 - xy[i, 1])
            if r[i] > wall_min:
                r[i] = wall_min
            r[i] = max(r[i], 0.001)
    return xy, r


def _anneal_from_start(xy, r, deadline, seed):
    """
    Multi-restart lambda annealing: start with high penalty (force feasibility),
    then reduce penalty to let radii grow.
    """
    n = N
    v = np.concatenate([xy.ravel(), r])
    rng = np.random.RandomState(seed)

    # Annealing schedule: start high, decrease
    beta = 30.0
    lam_start = 500.0
    lam_end = 10.0
    n_stages = 6

    for stage in range(n_stages):
        if time.time() > deadline - 2:
            break
        frac = stage / max(n_stages - 1, 1)
        lam = lam_start * (lam_end / lam_start) ** frac
        # Increase beta slightly as we refine
        beta_cur = beta + 10.0 * frac

        v = _lbfgs_step(v, n, beta_cur, lam, deadline,
                        max_iter=1500, maxfun=8000)

        # Extract
        x = v[0:n]
        y = v[n:2*n]
        r = v[2*n:3*n]
        xy = np.column_stack([x, y])

        # Feasibility projection to clean up
        xy, r = _feasibility_projection(xy, r, iters=20)
        v = np.concatenate([xy.ravel(), r])

    # Final polish with low penalty
    if time.time() < deadline - 2:
        v = _lbfgs_step(v, n, 60.0, 5.0, deadline, max_iter=2000, maxfun=10000)
        x = v[0:n]
        y = v[n:2*n]
        r = v[2*n:3*n]
        xy = np.column_stack([x, y])
        xy, r = _feasibility_projection(xy, r, iters=30)

    return xy, r


def _high_precision_polish(xy, r, deadline):
    """Final high-precision polish."""
    n = N
    v0 = np.concatenate([xy.ravel(), r])
    bounds = []
    for i in range(n):
        bounds.append((0.0001, 0.9999))
    for i in range(n):
        bounds.append((0.0001, 0.9999))
    for i in range(n):
        bounds.append((0.0001, 0.4999))

    def obj(v):
        if time.time() > deadline:
            return float('inf')
        return _smooth_objective(v, n, 80.0, 3.0)

    def grad(v):
        if time.time() > deadline:
            return np.zeros(3 * n)
        return _smooth_grad(v, n, 80.0, 3.0)

    try:
        result = minimize(obj, v0, jac=grad, method='L-BFGS-B',
                          bounds=bounds,
                          options={'maxiter': 3000, 'ftol': 1e-12,
                                   'gtol': 1e-9, 'maxfun': 50000})
        v_opt = result.x
        new_xy = np.column_stack([v_opt[0:n], v_opt[n:2*n]])
        new_r = v_opt[2*n:3*n]
        total, valid = _eval_packing(new_xy, new_r)
        if valid:
            return new_xy, new_r
    except Exception:
        pass
    return xy, r


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    best_xy = None
    best_r = None
    best_total = 0.0
    best_valid = False

    def _update(xy, r):
        nonlocal best_xy, best_r, best_total, best_valid
        total, valid = _eval_packing(xy, r)
        if valid and total > best_total:
            best_total = total
            best_valid = True
            best_xy = xy.copy()
            best_r = r.copy()

    # Phase 1: Hexagonal seeds with annealing
    hex_configs = [
        (0, 0.0, 0.78),
        (1, 0.0, 0.82),
        (2, 0.02, 0.75),
        (3, 0.0, 0.85),
        (4, 0.03, 0.72),
    ]
    for seed, jitter, scale in hex_configs:
        if time.time() > deadline - 5:
            break
        xy, r = _hex_init(N, seed=seed, jitter=jitter, scale=scale)
        out_xy, out_r = _anneal_from_start(xy, r, deadline, seed=seed)
        _update(out_xy, out_r)

    # Phase 2: Random seeds with annealing
    for seed in range(6):
        if time.time() > deadline - 5:
            break
        xy, r = _random_init(N, seed=seed)
        out_xy, out_r = _anneal_from_start(xy, r, deadline, seed=100 + seed)
        _update(out_xy, out_r)

    # Phase 3: High-precision polish from best
    if best_valid and best_xy is not None and time.time() < deadline - 3:
        hp_xy, hp_r = _high_precision_polish(best_xy.copy(), best_r.copy(), deadline)
        _update(hp_xy, hp_r)

    # Fallback
    if best_xy is None or not best_valid:
        side = int(np.ceil(np.sqrt(N)))
        result = np.zeros((N, 3))
        for i in range(N):
            gx = i % side
            gy = i // side
            result[i, 0] = (gx + 0.5) / side
            result[i, 1] = (gy + 0.5) / side
            result[i, 2] = 0.4 / side
        return result

    # Final validation and clipping
    n = len(best_xy)
    result = np.zeros((n, 3))
    for i in range(n):
        x, y, ri = best_xy[i, 0], best_xy[i, 1], best_r[i]
        ri = min(ri, x, 1.0 - x, y, 1.0 - y)
        ri = max(ri, 0.001)
        result[i, 2] = ri
        result[i, 0] = np.clip(x, ri, 1.0 - ri)
        result[i, 1] = np.clip(y, ri, 1.0 - ri)

    # Final check
    final_xy = result[:, :2]
    final_r = result[:, 2]
    total, valid = _eval_packing(final_xy, final_r)
    if not valid:
        side = int(np.ceil(np.sqrt(N)))
        result = np.zeros((N, 3))
        for i in range(N):
            gx = i % side
            gy = i // side
            result[i, 0] = (gx + 0.5) / side
            result[i, 1] = (gy + 0.5) / side
            result[i, 2] = 0.4 / side

    return result
# EVOLVE-BLOCK-END
