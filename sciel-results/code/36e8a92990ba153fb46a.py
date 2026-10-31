"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

# ---------- objective / constraint helpers ----------

def _unpack(x):
    pos = x[:2*N].reshape(N, 2)
    radii = np.maximum(x[2*N:], 1e-8)
    return pos, radii

def _objective(x, penalty_weight=1000.0):
    """minimize -sum(radii) + penalty * squared violations (for L-BFGS-B)."""
    pos, radii = _unpack(x)

    bv = (np.maximum(0.0, radii - pos[:, 0]) +
          np.maximum(0.0, pos[:, 0] - (1.0 - radii)) +
          np.maximum(0.0, radii - pos[:, 1]) +
          np.maximum(0.0, pos[:, 1] - (1.0 - radii)))
    boundary_pen = np.sum(bv * bv)

    dx = pos[:, 0, None] - pos[None, :, 0]
    dy = pos[:, 1, None] - pos[None, :, 1]
    dist_sq = dx * dx + dy * dy
    min_dist = radii[:, None] + radii[None, :]
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    viol = np.maximum(0.0, min_dist * min_dist - dist_sq)
    overlap_pen = np.sum(viol[mask] * viol[mask])

    return -np.sum(radii) + penalty_weight * (boundary_pen + overlap_pen)

def _constraint_overlap(x):
    """r_i + r_j - dist_ij  (<= 0 means no overlap)."""
    pos, radii = _unpack(x)
    dx = pos[:, 0, None] - pos[None, :, 0]
    dy = pos[:, 1, None] - pos[None, :, 1]
    dist = np.sqrt(dx * dx + dy * dy + 1e-12)
    min_dist = radii[:, None] + radii[None, :]
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    return min_dist[mask] - dist[mask]

def _constraint_boundary(x):
    """radius - dist_to_wall  (<= 0 means inside)."""
    pos, radii = _unpack(x)
    d = np.stack([pos[:, 0], 1.0 - pos[:, 0], pos[:, 1], 1.0 - pos[:, 1]], axis=1)
    soft_min = d.min(axis=1)
    return radii - soft_min

def _is_valid(pos, radii, tol=1e-7):
    n = len(radii)
    if np.any(pos[:, 0] - radii < -tol) or np.any(pos[:, 0] + radii > 1.0 + tol):
        return False
    if np.any(pos[:, 1] - radii < -tol) or np.any(pos[:, 1] + radii > 1.0 + tol):
        return False
    dx = pos[:, 0, None] - pos[None, :, 0]
    dy = pos[:, 1, None] - pos[None, :, 1]
    dist_sq = dx * dx + dy * dy
    min_dist = radii[:, None] + radii[None, :]
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    if np.any(dist_sq[mask] < (min_dist[mask] - tol) ** 2):
        return False
    return True

def _post_process(pos, radii, iters=200):
    """Vectorized shrink to guarantee strict validity."""
    pos = pos.copy().astype(float)
    radii = radii.copy().astype(float)
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    ii, jj = np.nonzero(mask)

    for _ in range(iters):
        valid = True
        # boundaries
        maxr_wall = np.minimum(np.minimum(pos[:, 0], 1.0 - pos[:, 0]),
                               np.minimum(pos[:, 1], 1.0 - pos[:, 1]))
        over = radii > maxr_wall
        if np.any(over):
            radii[over] = np.maximum(maxr_wall[over] * 0.9999, 1e-5)
            valid = False
        # overlaps
        dx = pos[ii, 0] - pos[jj, 0]
        dy = pos[ii, 1] - pos[jj, 1]
        dist = np.sqrt(dx * dx + dy * dy)
        sr = radii[ii] + radii[jj]
        bad = dist < sr
        if np.any(bad):
            k = (dist[bad] / (sr[bad] + 1e-12)) * 0.9999
            idx_i = ii[bad]
            idx_j = jj[bad]
            # for each bad circle take the min shrink factor it is involved in
            for a, b in zip(idx_i, idx_j):
                kk = k[list(idx_i).index(a)] if a in idx_i else 1.0
            # simpler: apply per-pair
            for p in range(len(idx_i)):
                a, b = idx_i[p], idx_j[p]
                radii[a] = max(radii[a] * k[p], 1e-5)
                radii[b] = max(radii[b] * k[p], 1e-5)
            valid = False
        if valid:
            break
    return pos, radii

def _grow_radii(pos, radii, deadline, max_iter=300):
    """Iteratively inflate radii to the local maximum given fixed neighbours."""
    pos = pos.copy().astype(float)
    radii = radii.copy().astype(float)
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        improved = False
        maxr_wall = np.minimum(np.minimum(pos[:, 0], 1.0 - pos[:, 0]),
                               np.minimum(pos[:, 1], 1.0 - pos[:, 1]))
        # per-circle max from neighbours
        dx = pos[:, 0, None] - pos[None, :, 0]
        dy = pos[:, 1, None] - pos[None, :, 1]
        dist = np.sqrt(dx * dx + dy * dy)
        np.fill_diagonal(dist, np.inf)
        maxr_neigh = np.min(dist - radii[None, :], axis=1)
        maxr = np.minimum(maxr_wall, maxr_neigh)
        new_r = np.clip(maxr * 0.9999, 1e-5, 0.5)
        grow = new_r > radii + 1e-8
        if np.any(grow):
            radii[grow] = new_r[grow]
            improved = True
        if not improved:
            break
    pos, radii = _post_process(pos, radii)
    return pos, radii

# ---------- initializations ----------

def _init_hexagonal():
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    r = 0.11
    dx = r * np.sqrt(3)
    dy = r * 1.5
    k = 0
    for row in range(6):
        offset = dx / 2 if row % 2 == 1 else 0
        start_x = r + offset
        y = row * dy + r
        for col in range(5):
            if k >= N:
                break
            x = start_x + col * dx
            if x + r <= 1.0 and x - r >= 0.0 and y + r <= 1.0 and y - r >= 0.0:
                pos[k] = [x, y]
                radii[k] = r
                k += 1
        if k >= N:
            break
    while k < N:
        rng = np.random.RandomState(k * 7 + 13)
        placed = False
        for _ in range(100):
            x = rng.uniform(0.1, 0.9)
            y = rng.uniform(0.1, 0.9)
            min_dist = min(x, 1 - x, y, 1 - y)
            for j in range(k):
                d = np.sqrt((x - pos[j, 0]) ** 2 + (y - pos[j, 1]) ** 2)
                min_dist = min(min_dist, d - radii[j])
            if min_dist > 0.02:
                pos[k] = [x, y]
                radii[k] = min_dist * 0.95
                k += 1
                placed = True
                break
        if not placed:
            pos[k] = [0.5, 0.5]
            radii[k] = 0.01
            k += 1
    return pos, radii

def _init_corner_biased(seed=7):
    rng = np.random.RandomState(seed)
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    corners = [(0.15, 0.15), (0.85, 0.15), (0.15, 0.85), (0.85, 0.85)]
    for i, (cx, cy) in enumerate(corners):
        pos[i] = [cx + rng.uniform(-0.03, 0.03), cy + rng.uniform(-0.03, 0.03)]
        radii[i] = 0.18
    for i in range(4, N):
        if rng.random() < 0.4:
            edge = rng.randint(4)
            t = rng.uniform(0.1, 0.9)
            if edge == 0:
                x, y = t, 0.08
            elif edge == 1:
                x, y = t, 0.92
            elif edge == 2:
                x, y = 0.08, t
            else:
                x, y = 0.92, t
        else:
            x, y = rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9)
        max_r = min(x, 1.0 - x, y, 1.0 - y)
        for j in range(i):
            dx = x - pos[j, 0]
            dy = y - pos[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            max_r = min(max_r, dist - radii[j])
        if max_r < 0.01:
            for attempt in range(200):
                x2 = rng.uniform(0.05, 0.95)
                y2 = rng.uniform(0.05, 0.95)
                max_r2 = min(x2, 1.0 - x2, y2, 1.0 - y2)
                for j in range(i):
                    dx = x2 - pos[j, 0]
                    dy = y2 - pos[j, 1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    max_r2 = min(max_r2, dist - radii[j])
                if max_r2 > max_r:
                    max_r = max_r2
                    x, y = x2, y2
        pos[i] = [x, y]
        radii[i] = max(max_r, 0.01) * 0.95
    return pos, radii

def _init_levelset(seed=3):
    """Greedy placement at local maxima of a distance field (FMM-style)."""
    rng = np.random.RandomState(seed)
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    grid = np.linspace(0.0, 1.0, 61)
    gx, gy = np.meshgrid(grid, grid)
    # distance to square boundary
    field = np.minimum(np.minimum(gx, 1.0 - gx), np.minimum(gy, 1.0 - gy))
    centers = []
    radii_list = []
    for i in range(N):
        # subtract influence of placed circles
        for (cx, cy, cr) in centers:
            d = np.sqrt((gx - cx) ** 2 + (gy - cy) ** 2)
            field = np.minimum(field, d - cr)
        # find a good local max
        best_val = -1
        best_xy = (0.5, 0.5)
        # sample candidates around the global max plus random jitter
        flat_idx = np.argmax(field)
        bx, by = grid[flat_idx // 61], grid[flat_idx % 61]
        candidates = [(bx, by)]
        for _ in range(40):
            candidates.append((rng.uniform(0.02, 0.98), rng.uniform(0.02, 0.98)))
        for (cx, cy) in candidates:
            val = min(cx, 1.0 - cx, cy, 1.0 - cy)
            for (px, py, pr) in centers:
                val = min(val, np.sqrt((cx - px) ** 2 + (cy - py) ** 2) - pr)
            if val > best_val:
                best_val = val
                best_xy = (cx, cy)
        r = max(best_val, 0.005) * 0.98
        pos[i] = best_xy
        radii[i] = r
        centers.append((best_xy[0], best_xy[1], r))
        radii_list.append(r)
    return pos, radii

def _init_random(seed):
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.1, 0.9, (N, 2))
    radii = rng.uniform(0.04, 0.12, N)
    return _post_process(pos, radii)

# ---------- optimization ----------

_BOUNDS = [(0.01, 0.99), (0.01, 0.99)] * N + [(0.01, 0.5)] * N

def _slsqp_pass(x, deadline, maxiter=400):
    """SLSQP with hard equality/inequality constraints -> strict feasibility."""
    cons = [{'type': 'ineq', 'fun': lambda v: -_constraint_overlap(v)},
            {'type': 'ineq', 'fun': lambda v: -_constraint_boundary(v)}]
    try:
        res = minimize(_objective, x, method='SLSQP', bounds=_BOUNDS,
                       args=(1000.0), constraints=cons,
                       options={'maxiter': maxiter, 'ftol': 1e-12})
        if res.fun < _objective(x, 1000.0) - 1e-10:
            return res.x
    except Exception:
        pass
    return x

def _lbfgs_pass(x, deadline, maxiter=400):
    try:
        res = minimize(_objective, x, method='L-BFGS-B', bounds=_BOUNDS,
                       args=(5000.0),
                       options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8})
        if res.fun < _objective(x, 5000.0) - 1e-10:
            return res.x
    except Exception:
        pass
    return x

def _simulated_annealing(pos, radii, n_steps=1500, t0=0.03, t_end=0.0005,
                         seed=0, deadline=None):
    rng = np.random.RandomState(seed)
    pos = pos.copy().astype(float)
    radii = radii.copy().astype(float)
    best_pos, best_radii = pos.copy(), radii.copy()
    best_score = np.sum(radii)
    cur = _objective(np.concatenate([pos.flatten(), radii]), 500.0)
    for step in range(n_steps):
        if deadline and time.time() > deadline:
            break
        frac = step / max(n_steps - 1, 1)
        temp = t0 * (t_end / t0) ** frac
        i = rng.randint(N)
        nx = np.clip(pos[i, 0] + rng.normal(0, temp * 0.02), 0.01, 0.99)
        ny = np.clip(pos[i, 1] + rng.normal(0, temp * 0.02), 0.01, 0.99)
        nr = np.clip(radii[i] + rng.normal(0, temp * 0.01), 0.01, 0.4)
        ox, oy, orr = pos[i, 0], pos[i, 1], radii[i]
        pos[i, 0], pos[i, 1], radii[i] = nx, ny, nr
        new = _objective(np.concatenate([pos.flatten(), radii]), 500.0)
        if new - cur < 0 or rng.random() < np.exp(-(new - cur) / max(temp, 1e-8)):
            cur = new
        else:
            pos[i, 0], pos[i, 1], radii[i] = ox, oy, orr
        if step % 60 == 0:
            pp, rr = _post_process(pos, radii)
            if _is_valid(pp, rr):
                s = np.sum(rr)
                if s > best_score:
                    best_score, best_pos, best_radii = s, pp.copy(), rr.copy()
    return best_pos, best_radii

# ---------- driver ----------

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    inits = [
        _init_hexagonal(),
        _init_corner_biased(seed=7),
        _init_levelset(seed=3),
        _init_random(100),
    ]

    best_pos, best_radii = inits[0]
    best_score = np.sum(best_radii)

    for idx, (pos_init, rad_init) in enumerate(inits):
        if time.time() > deadline - 20:
            break
        x = np.concatenate([pos_init.flatten(), rad_init])

        # 1) SLSQP hard-constraint pass (the structural idea)
        if time.time() < deadline - 12:
            x = _slsqp_pass(x, deadline)
        # 2) L-BFGS-B polish
        if time.time() < deadline - 10:
            x = _lbfgs_pass(x, deadline)

        pos_opt = x[:2*N].reshape(N, 2)
        radii_opt = np.maximum(x[2*N:], 1e-4)
        pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
        if _is_valid(pos_pp, radii_pp):
            s = np.sum(radii_pp)
            if s > best_score:
                best_score, best_pos, best_radii = s, pos_pp.copy(), radii_pp.copy()

        # 3) radius growth (the refinement machinery)
        if time.time() < deadline - 10:
            pos_g, rad_g = _grow_radii(pos_pp, radii_pp, deadline - 8)
            s = np.sum(rad_g)
            if _is_valid(pos_g, rad_g) and s > best_score:
                best_score, best_pos, best_radii = s, pos_g.copy(), rad_g.copy()

        # 4) simulated annealing to escape local minima
        if time.time() < deadline - 8:
            try:
                pos_sa, rad_sa = _simulated_annealing(
                    pos_pp, radii_pp, n_steps=1200, t0=0.03, t_end=0.0005,
                    seed=idx * 100, deadline=deadline - 6)
                pos_pp2, rad_pp2 = _post_process(pos_sa, rad_sa)
                pos_g2, rad_g2 = _grow_radii(pos_pp2, rad_pp2, deadline - 5)
                s = np.sum(rad_g2)
                if _is_valid(pos_g2, rad_g2) and s > best_score:
                    best_score, best_pos, best_radii = s, pos_g2.copy(), rad_g2.copy()
            except Exception:
                pass

    # final global polish on best
    if time.time() < deadline - 6:
        x = np.concatenate([best_pos.flatten(), best_radii])
        x = _slsqp_pass(x, deadline, maxiter=300)
        pos_opt = x[:2*N].reshape(N, 2)
        radii_opt = np.maximum(x[2*N:], 1e-4)
        pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
        pos_g, rad_g = _grow_radii(pos_pp, radii_pp, deadline - 3)
        s = np.sum(rad_g)
        if _is_valid(pos_g, rad_g) and s > best_score:
            best_score, best_pos, best_radii = s, pos_g.copy(), rad_g.copy()

    # final validation / fallback
    best_pos, best_radii = _post_process(best_pos, best_radii)
    if not _is_valid(best_pos, best_radii):
        pos = np.zeros((N, 2))
        radii = np.zeros(N)
        k = 0
        for i in range(5):
            for j in range(5):
                if k < N - 1:
                    pos[k] = [(i + 0.5) / 5.0, (j + 0.5) / 5.0]
                    radii[k] = 0.095
                    k += 1
        pos[k] = [0.5, 0.02]
        radii[k] = 0.01
        best_pos, best_radii = _post_process(pos, radii)

    return np.column_stack([best_pos, best_radii])

# EVOLVE-BLOCK-END
