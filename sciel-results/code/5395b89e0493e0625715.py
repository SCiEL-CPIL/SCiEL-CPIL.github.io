"""Circle packing in unit square: maximize sum of radii of 26 disjoint circles in [0,1]^2."""
import time
import numpy as np
from scipy.optimize import minimize

N = 26
_DEADLINE = 150.0
_START = time.time()
_EPS = 1e-9


def _remaining():
    return _DEADLINE - (time.time() - _START) - 5.0  # 5s safety margin


def _feasible_score(pos, rad):
    """Return (value, feasible). value = sum(rad) if feasible else -inf."""
    n = len(rad)
    # wall check
    if np.any(rad < -_EPS) or np.any(pos[:, 0] - rad < -_EPS) or np.any(pos[:, 0] + rad > 1 + _EPS) \
       or np.any(pos[:, 1] - rad < -_EPS) or np.any(pos[:, 1] + rad > 1 + _EPS):
        return -np.inf, False
    # pairwise
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            d2 = dx * dx + dy * dy
            rr = rad[i] + rad[j]
            if d2 < rr * rr - _EPS:
                return -np.inf, False
    return float(np.sum(rad)), True


def _project_feasible(pos, rad, iters=40):
    """Heuristic repair: shrink radii / nudge to satisfy constraints. Returns (pos, rad, feasible)."""
    pos = pos.copy()
    rad = rad.copy()
    n = len(rad)
    for _ in range(iters):
        # clamp to walls by shrinking radius and centering
        for i in range(n):
            x, y, r = pos[i, 0], pos[i, 1], rad[i]
            # wall constraints
            r = min(r, x, 1.0 - x, y, 1.0 - y)
            r = max(r, 0.0)
            # nudge center to be inside if it was outside
            x = min(max(x, r), 1.0 - r)
            y = min(max(y, r), 1.0 - r)
            pos[i, 0], pos[i, 1], rad[i] = x, y, r
        # resolve overlaps by shrinking the smaller of the two radii
        changed = False
        for _ in range(3):
            inner_changed = False
            for i in range(n):
                for j in range(i + 1, n):
                    dx = pos[i, 0] - pos[j, 0]
                    dy = pos[i, 1] - pos[j, 1]
                    d = np.sqrt(dx * dx + dy * dy)
                    need = rad[i] + rad[j]
                    if d < need - _EPS:
                        # shrink both proportionally to their radii
                        a = rad[i]
                        b = rad[j]
                        # scale factor
                        s = (d - _EPS) / (need + _EPS) if need > 0 else 1.0
                        rad[i] = max(a * s, 0.0)
                        rad[j] = max(b * s, 0.0)
                        inner_changed = True
            if not inner_changed:
                break
            changed = True
        if not changed:
            break
    val, feas = _feasible_score(pos, rad)
    return pos, rad, feas


def _soft_objective(z, lam):
    """z = [x0,y0,r0, x1,y1,r1, ...]. Minimize -sum(r) + lam*(overlap + wall penalties)."""
    n = N
    x = z[0::3]
    y = z[1::3]
    r = z[2::3]
    base = -np.sum(r)
    pen = 0.0
    # wall penalties (smooth)
    pen += np.sum(np.maximum(r - x, 0.0) ** 2)
    pen += np.sum(np.maximum(x + r - 1.0, 0.0) ** 2)
    pen += np.sum(np.maximum(r - y, 0.0) ** 2)
    pen += np.sum(np.maximum(y + r - 1.0, 0.0) ** 2)
    # pairwise overlap - vectorized
    # Build pairwise difference matrices (n x n)
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d2 = dx * dx + dy * dy
    need = r[:, None] + r[None, :]
    # Only upper triangle
    iu = np.triu_indices(n, 1)
    d2u = d2[iu]
    needu = need[iu]
    # Find overlapping pairs
    mask = d2u < needu * needu
    if np.any(mask):
        du = np.sqrt(np.maximum(d2u[mask], 1e-12))
        pen += np.sum((needu[mask] - du) ** 2)
    return base + lam * pen


def _soft_grad(z, lam):
    n = N
    x = z[0::3].copy()
    y = z[1::3].copy()
    r = z[2::3].copy()
    gx = np.zeros(n)
    gy = np.zeros(n)
    gr = np.zeros(n)
    # base: d(-sum r)/dr = -1
    gr -= 1.0
    # wall: term (r-x)^2 when r>x
    m = r - x
    mpos = m > 0
    if np.any(mpos):
        gx[mpos] += lam * 2.0 * m[mpos] * (-1.0)
        gr[mpos] += lam * 2.0 * m[mpos]
    m = x + r - 1.0
    mpos = m > 0
    if np.any(mpos):
        gx[mpos] += lam * 2.0 * m[mpos]
        gr[mpos] += lam * 2.0 * m[mpos]
    m = r - y
    mpos = m > 0
    if np.any(mpos):
        gy[mpos] += lam * 2.0 * m[mpos] * (-1.0)
        gr[mpos] += lam * 2.0 * m[mpos]
    m = y + r - 1.0
    mpos = m > 0
    if np.any(mpos):
        gy[mpos] += lam * 2.0 * m[mpos]
        gr[mpos] += lam * 2.0 * m[mpos]
    # pairwise - vectorized
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d2 = dx * dx + dy * dy
    need = r[:, None] + r[None, :]
    iu = np.triu_indices(n, 1)
    d2u = d2[iu]
    needu = need[iu]
    mask = d2u < needu * needu
    if np.any(mask):
        idx_i = iu[0][mask]
        idx_j = iu[1][mask]
        dxu = dx[iu][mask]
        dyu = dy[iu][mask]
        du = np.sqrt(np.maximum(d2u[mask], 1e-12))
        fu = needu[mask] - du  # > 0
        inv_u = 1.0 / du
        # Accumulate gradients using np.add.at for repeated indices
        np.add.at(gx, idx_i, lam * 2.0 * fu * (-dxu * inv_u))
        np.add.at(gx, idx_j, lam * 2.0 * fu * (dxu * inv_u))
        np.add.at(gy, idx_i, lam * 2.0 * fu * (-dyu * inv_u))
        np.add.at(gy, idx_j, lam * 2.0 * fu * (dyu * inv_u))
        np.add.at(gr, idx_i, lam * 2.0 * fu)
        np.add.at(gr, idx_j, lam * 2.0 * fu)
    out = np.empty(3 * n)
    out[0::3] = gx
    out[1::3] = gy
    out[2::3] = gr
    return out


def _hex_seed(scale=1.0, jitter=0.0, rng=None):
    """Generate a hexagonal-ish arrangement scaled to fit, with optional jitter."""
    rng = rng or np.random.default_rng()
    pts = []
    rows = 0
    # build hex lattice until we have >= N points
    while len(pts) < N:
        row = rows
        y = row * 0.8660254
        offset = (0.5 if row % 2 else 0.0)
        col = 0
        while len(pts) < N:
            x = col + offset
            pts.append((x, y))
            col += 1
        rows += 1
    pts = np.array(pts[:N])
    # normalize to unit square with margin
    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)
    span = max(xmax - xmin, ymax - ymin, 1e-9)
    pts = (pts - [xmin, ymin]) / span
    # center in [0,1]
    cx = (xmax - xmin) / span
    cy = (ymax - ymin) / span
    pts[:, 0] = 0.5 - cx / 2 + pts[:, 0] * cx
    pts[:, 1] = 0.5 - cy / 2 + pts[:, 1] * cy
    if jitter > 0:
        pts = pts + rng.normal(0, jitter, size=pts.shape)
    # radius: use min distance to neighbor /2 and to walls
    rad = np.full(N, 0.2)
    for i in range(N):
        dmin = min(pts[i, 0], 1 - pts[i, 0], pts[i, 1], 1 - pts[i, 1])
        for j in range(N):
            if i != j:
                dmin = min(dmin, np.hypot(pts[i, 0] - pts[j, 0], pts[i, 1] - pts[j, 1]))
        rad[i] = 0.5 * dmin * 0.98
    return pts, rad


def _repulsion_relax(pos, iters=300, rng=None):
    """Spread points apart via pairwise repulsion + wall attraction. Returns relaxed pos."""
    rng = rng or np.random.default_rng()
    n = len(pos)
    pos = pos.copy()
    # target spacing
    target = 1.0 / np.sqrt(n) * 1.1
    for it in range(iters):
        grad = np.zeros_like(pos)
        for i in range(n):
            for j in range(i + 1, n):
                d = pos[i] - pos[j]
                dist = np.linalg.norm(d)
                if dist < 1e-8:
                    d = rng.normal(0, 0.01, 2)
                    dist = np.linalg.norm(d)
                # repulsion if closer than target
                if dist < target:
                    f = (target - dist) / dist
                    grad[i] += f * d
                    grad[j] -= f * d
        # wall attraction
        for i in range(n):
            if pos[i, 0] < 0.05:
                grad[i, 0] += (0.05 - pos[i, 0]) * 5.0
            elif pos[i, 0] > 0.95:
                grad[i, 0] -= (pos[i, 0] - 0.95) * 5.0
            if pos[i, 1] < 0.05:
                grad[i, 1] += (0.05 - pos[i, 1]) * 5.0
            elif pos[i, 1] > 0.95:
                grad[i, 1] -= (pos[i, 1] - 0.95) * 5.0
        pos += 0.02 * grad
        pos = np.clip(pos, 0.03, 0.97)
    return pos


def _grow_radii(pos, rad, time_budget=20.0):
    """Polish: maximize sum(rad) with fixed positions, using smooth max-penalty.
    Returns (rad, feasible)."""
    n = N
    pos = np.asarray(pos)
    rad = np.asarray(rad, dtype=float).copy()
    t0 = time.time()
    # Precompute pairwise distances and wall distances
    dx = pos[:, 0:1] - pos[:, 0:1].T
    dy = pos[:, 1:2] - pos[:, 1:2].T
    dist_mat = np.sqrt(dx * dx + dy * dy)
    np.fill_diagonal(dist_mat, np.inf)
    # wall distances per circle
    wall_d = np.minimum(np.minimum(pos[:, 0], 1 - pos[:, 0]),
                        np.minimum(pos[:, 1], 1 - pos[:, 1]))

    def obj(r):
        # smooth max over pairwise constraints: r_i + r_j - dist_ij
        # and wall: r_i - wall_d_i
        pen = 0.0
        # pairwise
        for i in range(n):
            for j in range(i + 1, n):
                v = r[i] + r[j] - dist_mat[i, j]
                if v > 0:
                    pen += v * v
        # wall
        for i in range(n):
            v = r[i] - wall_d[i]
            if v > 0:
                pen += v * v
        return -np.sum(r) + pen

    def grad(r):
        g = -np.ones(n)
        for i in range(n):
            for j in range(i + 1, n):
                v = r[i] + r[j] - dist_mat[i, j]
                if v > 0:
                    g[i] += 2.0 * v
                    g[j] += 2.0 * v
        for i in range(n):
            v = r[i] - wall_d[i]
            if v > 0:
                g[i] += 2.0 * v
        return g

    # Start from current radii, slightly shrunk for safety
    r0 = np.maximum(rad * 0.95, 0.01)
    # Iteratively grow: increase a base radius and optimize
    best_r = r0.copy()
    best_val = float(np.sum(r0))
    # Try growing by scaling
    for scale in [1.0, 1.1, 1.25, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0]:
        if time.time() - t0 > time_budget:
            break
        r_init = np.minimum(wall_d * 0.98, 0.4) * scale * 0.5
        r_init = np.clip(r_init, 0.01, wall_d * 0.99)
        res = minimize(obj, r_init, jac=grad, method='L-BFGS-B',
                       bounds=[(0.01, wall_d[i] * 0.999) for i in range(n)],
                       options={'maxiter': 200, 'gtol': 1e-8, 'ftol': 1e-10})
        r_opt = np.clip(res.x, 0.01, wall_d * 0.999)
        # Verify feasibility
        feas = True
        for i in range(n):
            if r_opt[i] > wall_d[i] + 1e-9:
                feas = False
                break
        if feas:
            for i in range(n):
                for j in range(i + 1, n):
                    if r_opt[i] + r_opt[j] > dist_mat[i, j] + 1e-9:
                        feas = False
                        break
                if not feas:
                    break
        if feas:
            val = float(np.sum(r_opt))
            if val > best_val:
                best_val = val
                best_r = r_opt.copy()
    return best_r, True


def construct():
    global _START
    _START = time.time()
    rng = np.random.default_rng(12345)

    best_val = -np.inf
    best_pos = None
    best_rad = None

    # Seed set: a few hex variants + random + repulsion-relaxed
    seeds = []
    for jitter in [0.0, 0.02, 0.05, 0.1]:
        p, r = _hex_seed(jitter=jitter, rng=rng)
        seeds.append((p, r))
    # random seeds
    for _ in range(4):
        p = rng.uniform(0.15, 0.85, size=(N, 2))
        r = rng.uniform(0.05, 0.12, size=N)
        seeds.append((p, r))
    # repulsion-relaxed seeds
    for _ in range(3):
        p = rng.uniform(0.1, 0.9, size=(N, 2))
        p = _repulsion_relax(p, iters=200, rng=rng)
        # compute initial radii
        rad = np.zeros(N)
        for i in range(N):
            dmin = min(p[i, 0], 1 - p[i, 0], p[i, 1], 1 - p[i, 1])
            for j in range(N):
                if i != j:
                    dmin = min(dmin, np.hypot(p[i, 0] - p[j, 0], p[i, 1] - p[j, 1]))
            rad[i] = 0.5 * dmin * 0.95
        seeds.append((p, rad))

    bounds = [(0, 1), (0, 1), (1e-4, 1)] * N

    def _try_seed(pos, rad, lam, max_iter):
        nonlocal best_val, best_pos, best_rad
        z = np.empty(3 * N)
        z[0::3] = pos[:, 0]
        z[1::3] = pos[:, 1]
        z[2::3] = rad
        res = minimize(_soft_objective, z, jac=_soft_grad, method='L-BFGS-B',
                       bounds=bounds, args=(lam,),
                       options={'maxiter': max_iter, 'gtol': 1e-6, 'ftol': 1e-8})
        z = res.x
        p = np.column_stack([z[0::3], z[1::3]])
        r = z[2::3]
        # project to feasibility
        p, r, feas = _project_feasible(p, r)
        if feas:
            val = float(np.sum(r))
            if val > best_val:
                best_val = val
                best_pos = p.copy()
                best_rad = r.copy()

    # Anneal lambda: start high (strong penalty) then lower to grow radii
    lams = [50.0, 20.0, 8.0, 3.0, 1.0, 0.4, 0.15]
    max_iters = 300

    for si, (pos, rad) in enumerate(seeds):
        if _remaining() < 20.0:
            break
        # run a short anneal per seed
        cur_pos, cur_rad = pos.copy(), rad.copy()
        for lam in lams:
            if _remaining() < 10.0:
                break
            # warm start from previous
            z = np.empty(3 * N)
            z[0::3] = cur_pos[:, 0]
            z[1::3] = cur_pos[:, 1]
            z[2::3] = cur_rad
            res = minimize(_soft_objective, z, jac=_soft_grad, method='L-BFGS-B',
                           bounds=bounds, args=(lam,),
                           options={'maxiter': max_iters, 'gtol': 1e-6, 'ftol': 1e-8})
            z = res.x
            cur_pos = np.column_stack([z[0::3], z[1::3]])
            cur_rad = z[2::3]
            cur_pos, cur_rad, feas = _project_feasible(cur_pos, cur_rad)
            if feas:
                val = float(np.sum(cur_rad))
                if val > best_val:
                    best_val = val
                    best_pos = cur_pos.copy()
                    best_rad = cur_rad.copy()

    # If we still have time, do extra random restarts from best
    if best_pos is not None and _remaining() > 30.0:
        for _ in range(12):
            if _remaining() < 10.0:
                break
            p = best_pos + rng.normal(0, 0.03, size=best_pos.shape)
            p = np.clip(p, 0.05, 0.95)
            r = best_rad * rng.uniform(0.7, 1.0, size=N)
            # quick anneal with a couple of lambdas
            cur_pos, cur_rad = p.copy(), r.copy()
            for lam in [10.0, 3.0, 0.5]:
                if _remaining() < 8.0:
                    break
                z = np.empty(3 * N)
                z[0::3] = cur_pos[:, 0]
                z[1::3] = cur_pos[:, 1]
                z[2::3] = cur_rad
                res = minimize(_soft_objective, z, jac=_soft_grad, method='L-BFGS-B',
                               bounds=bounds, args=(lam,),
                               options={'maxiter': 200, 'gtol': 1e-6, 'ftol': 1e-8})
                z = res.x
                cur_pos = np.column_stack([z[0::3], z[1::3]])
                cur_rad = z[2::3]
                cur_pos, cur_rad, feas = _project_feasible(cur_pos, cur_rad)
                if feas:
                    val = float(np.sum(cur_rad))
                    if val > best_val:
                        best_val = val
                        best_pos = cur_pos.copy()
                        best_rad = cur_rad.copy()

    # High-precision final polish: tighten tolerances and run more iterations
    if best_pos is not None and _remaining() > 15.0:
        for lam in [0.1, 0.05, 0.02, 0.01]:
            if _remaining() < 8.0:
                break
            z = np.empty(3 * N)
            z[0::3] = best_pos[:, 0]
            z[1::3] = best_pos[:, 1]
            z[2::3] = best_rad
            res = minimize(_soft_objective, z, jac=_soft_grad, method='L-BFGS-B',
                           bounds=bounds, args=(lam,),
                           options={'maxiter': 500, 'gtol': 1e-9, 'ftol': 1e-12})
            z = res.x
            p = np.column_stack([z[0::3], z[1::3]])
            r = z[2::3]
            p, r, feas = _project_feasible(p, r, iters=60)
            if feas:
                val = float(np.sum(r))
                if val > best_val:
                    best_val = val
                    best_pos = p.copy()
                    best_rad = r.copy()

    # Radius-growth polish: with fixed (or slightly adjusted) positions, maximize radii
    if best_pos is not None and _remaining() > 10.0:
        # Try growing radii from current best positions
        time_left = min(_remaining() - 5.0, 25.0)
        grown_rad, _ = _grow_radii(best_pos, best_rad, time_budget=time_left)
        # Verify the grown radii are feasible with best_pos
        feas = True
        for i in range(N):
            if grown_rad[i] < best_pos[i, 0] - 1e-9 or grown_rad[i] < 1 - best_pos[i, 0] - 1e-9 \
               or grown_rad[i] < best_pos[i, 1] - 1e-9 or grown_rad[i] < 1 - best_pos[i, 1] - 1e-9:
                feas = False
                break
        if feas:
            for i in range(N):
                for j in range(i + 1, N):
                    d = np.hypot(best_pos[i, 0] - best_pos[j, 0], best_pos[i, 1] - best_pos[j, 1])
                    if grown_rad[i] + grown_rad[j] > d + 1e-9:
                        feas = False
                        break
                if not feas:
                    break
        if feas:
            val = float(np.sum(grown_rad))
            if val > best_val:
                best_val = val
                best_rad = grown_rad.copy()
        # Also try: slightly perturb positions and grow
        if _remaining() > 10.0:
            for trial in range(4):
                if _remaining() < 8.0:
                    break
                p2 = best_pos + rng.normal(0, 0.01, size=best_pos.shape)
                p2 = np.clip(p2, 0.05, 0.95)
                time_left2 = min(_remaining() - 5.0, 10.0)
                grown_rad2, _ = _grow_radii(p2, best_rad, time_budget=time_left2)
                # Verify
                feas2 = True
                for i in range(N):
                    if grown_rad2[i] < p2[i, 0] - 1e-9 or grown_rad2[i] < 1 - p2[i, 0] - 1e-9 \
                       or grown_rad2[i] < p2[i, 1] - 1e-9 or grown_rad2[i] < 1 - p2[i, 1] - 1e-9:
                        feas2 = False
                        break
                if feas2:
                    for i in range(N):
                        for j in range(i + 1, N):
                            d = np.hypot(p2[i, 0] - p2[j, 0], p2[i, 1] - p2[j, 1])
                            if grown_rad2[i] + grown_rad2[j] > d + 1e-9:
                                feas2 = False
                                break
                        if not feas2:
                            break
                if feas2:
                    val2 = float(np.sum(grown_rad2))
                    if val2 > best_val:
                        best_val = val2
                        best_pos = p2.copy()
                        best_rad = grown_rad2.copy()

    # Fallback: guaranteed feasible seed if nothing found
    if best_pos is None:
        p, r = _hex_seed(jitter=0.0)
        p, r, feas = _project_feasible(p, r)
        if feas:
            best_pos, best_rad = p, r
        else:
            # absolute fallback: tiny circles in a grid
            side = int(np.ceil(np.sqrt(N)))
            cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                     for i in range(side) for j in range(side)]
            arr = np.array(cells[:N])
            return arr
        best_pos, best_rad = p, r

    out = np.column_stack([best_pos[:, 0], best_pos[:, 1], best_rad])
    return out


if __name__ == '__main__':
    res = construct()
    print("shape", res.shape)
    print("sum r", res[:, 2].sum())
    print("feasible", _feasible_score(res[:, :2], res[:, 2]))
