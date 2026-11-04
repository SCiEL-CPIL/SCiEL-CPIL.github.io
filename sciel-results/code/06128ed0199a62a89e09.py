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


def _langevin_dynamics(pos, rad, rng, t_max=30.0, dt=0.002, n_cool=2000):
    """
    Langevin dynamics for hard disks: particles repel when overlapping,
    walls reflect. Adaptive cooling reduces velocity over time to settle
    into a dense jammed state. Returns (pos, rad).
    """
    n = len(rad)
    pos = pos.astype(np.float64).copy()
    vel = np.zeros((n, 2))
    # Initial velocity from random kicks
    vel += rng.normal(0, 0.05, size=(n, 2))
    
    # Cooling schedule: T goes from T0 to T_end exponentially
    T0 = 0.05
    Tend = 0.0005
    
    for step in range(n_cool):
        if _remaining() < 5.0:
            break
        # Cooling factor
        alpha = step / max(n_cool - 1, 1)
        T = T0 * (Tend / T0) ** alpha
        
        # Compute forces
        force = np.zeros((n, 2))
        # Wall forces (soft repulsion)
        wall_k = 50.0
        for i in range(n):
            x, y, r = pos[i, 0], pos[i, 1], rad[i]
            if x < r:
                force[i, 0] += wall_k * (r - x)
            if x > 1.0 - r:
                force[i, 0] -= wall_k * (x - (1.0 - r))
            if y < r:
                force[i, 1] += wall_k * (r - y)
            if y > 1.0 - r:
                force[i, 1] -= wall_k * (y - (1.0 - r))
        
        # Pairwise repulsion (soft)
        pair_k = 100.0
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                d = np.sqrt(dx * dx + dy * dy)
                need = rad[i] + rad[j]
                if d < need and d > 1e-10:
                    overlap = need - d
                    f = pair_k * overlap
                    fx = f * dx / d
                    fy = f * dy / d
                    force[i, 0] += fx
                    force[i, 1] += fy
                    force[j, 0] -= fx
                    force[j, 1] -= fy
                elif d <= 1e-10:
                    # Random kick to separate
                    angle = rng.uniform(0, 2 * np.pi)
                    force[i, 0] += pair_k * 0.01 * np.cos(angle)
                    force[i, 1] += pair_k * 0.01 * np.sin(angle)
                    force[j, 0] -= pair_k * 0.01 * np.cos(angle)
                    force[j, 1] -= pair_k * 0.01 * np.sin(angle)
        
        # Langevin update: F = m*a, m=1, with damping and noise
        damping = 0.95
        noise_amp = np.sqrt(2 * T * dt)
        vel = vel * damping + force * dt + rng.normal(0, noise_amp, size=(n, 2))
        pos += vel * dt
        
        # Hard wall reflection
        for i in range(n):
            r = rad[i]
            if pos[i, 0] < r:
                pos[i, 0] = r
                vel[i, 0] = abs(vel[i, 0]) * 0.5
            if pos[i, 0] > 1.0 - r:
                pos[i, 0] = 1.0 - r
                vel[i, 0] = -abs(vel[i, 0]) * 0.5
            if pos[i, 1] < r:
                pos[i, 1] = r
                vel[i, 1] = abs(vel[i, 1]) * 0.5
            if pos[i, 1] > 1.0 - r:
                pos[i, 1] = 1.0 - r
                vel[i, 1] = -abs(vel[i, 1]) * 0.5
    
    return pos, rad


def _grow_radii(pos, rad, rng, max_grow=500, growth_step=0.001):
    """
    Gradually grow all radii while maintaining feasibility via small
    Langevin steps to resolve conflicts. Returns (pos, rad, feasible).
    """
    n = len(rad)
    pos = pos.copy()
    rad = rad.copy()
    current_growth = 0.0
    
    for _ in range(max_grow):
        if _remaining() < 5.0:
            break
        # Try to grow all radii by growth_step
        rad[:] += growth_step
        current_growth += growth_step
        
        # Check feasibility and repair if needed
        # Use a short Langevin burst to resolve overlaps
        for repair_iter in range(50):
            # Check for overlaps
            has_overlap = False
            for i in range(n):
                for j in range(i + 1, n):
                    dx = pos[i, 0] - pos[j, 0]
                    dy = pos[i, 1] - pos[j, 1]
                    d = np.sqrt(dx * dx + dy * dy)
                    need = rad[i] + rad[j]
                    if d < need - 1e-10:
                        has_overlap = True
                        break
                if has_overlap:
                    break
            
            # Check walls
            for i in range(n):
                if pos[i, 0] < rad[i] - 1e-10 or pos[i, 0] > 1.0 - rad[i] + 1e-10 or \
                   pos[i, 1] < rad[i] - 1e-10 or pos[i, 1] > 1.0 - rad[i] + 1e-10:
                    has_overlap = True
                    break
            
            if not has_overlap:
                break
            
            # Short Langevin step to resolve
            force = np.zeros((n, 2))
            wall_k = 50.0
            for i in range(n):
                x, y, r = pos[i, 0], pos[i, 1], rad[i]
                if x < r:
                    force[i, 0] += wall_k * (r - x)
                if x > 1.0 - r:
                    force[i, 0] -= wall_k * (x - (1.0 - r))
                if y < r:
                    force[i, 1] += wall_k * (r - y)
                if y > 1.0 - r:
                    force[i, 1] -= wall_k * (y - (1.0 - r))
            
            pair_k = 200.0
            for i in range(n):
                for j in range(i + 1, n):
                    dx = pos[i, 0] - pos[j, 0]
                    dy = pos[i, 1] - pos[j, 1]
                    d = np.sqrt(dx * dx + dy * dy)
                    need = rad[i] + rad[j]
                    if d < need and d > 1e-10:
                        overlap = need - d
                        f = pair_k * overlap
                        fx = f * dx / d
                        fy = f * dy / d
                        force[i, 0] += fx
                        force[i, 1] += fy
                        force[j, 0] -= fx
                        force[j, 1] -= fy
            
            # Small step
            dt = 0.001
            pos += force * dt
            # Clamp to walls
            for i in range(n):
                r = rad[i]
                pos[i, 0] = np.clip(pos[i, 0], r, 1.0 - r)
                pos[i, 1] = np.clip(pos[i, 1], r, 1.0 - r)
        
        # If we couldn't repair, shrink back
        val, feas = _feasible_score(pos, rad)
        if not feas:
            rad[:] -= growth_step
            current_growth -= growth_step
            # Try a slightly smaller growth
            growth_step *= 0.9
            if growth_step < 1e-5:
                break
    
    return pos, rad, _feasible_score(pos, rad)[1]


def construct():
    global _START
    _START = time.time()
    rng = np.random.default_rng(12345)

    best_val = -np.inf
    best_pos = None
    best_rad = None

    # Seed set: a few hex variants + random
    seeds = []
    for jitter in [0.0, 0.02, 0.05, 0.1]:
        p, r = _hex_seed(jitter=jitter, rng=rng)
        seeds.append((p, r))
    # random seeds
    for _ in range(6):
        p = rng.uniform(0.15, 0.85, size=(N, 2))
        r = rng.uniform(0.05, 0.12, size=N)
        seeds.append((p, r))

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

    # Langevin dynamics phase: use stochastic dynamics to find better packings
    if best_pos is not None and _remaining() > 40.0:
        for ld_trial in range(8):
            if _remaining() < 15.0:
                break
            # Start from best with perturbation
            p = best_pos + rng.normal(0, 0.05, size=best_pos.shape)
            p = np.clip(p, 0.05, 0.95)
            r = best_rad * rng.uniform(0.8, 1.0, size=N)
            # Run Langevin dynamics
            p_ld, r_ld = _langevin_dynamics(p, r, rng, t_max=10.0, dt=0.003, n_cool=1500)
            # Project to feasibility
            p_ld, r_ld, feas = _project_feasible(p_ld, r_ld)
            if feas:
                val = float(np.sum(r_ld))
                if val > best_val:
                    best_val = val
                    best_pos = p_ld.copy()
                    best_rad = r_ld.copy()
            # Try to grow radii from the Langevin result
            if feas and _remaining() > 12.0:
                p_g, r_g, feas_g = _grow_radii(p_ld, r_ld, rng, max_grow=300, growth_step=0.0008)
                if feas_g:
                    val_g = float(np.sum(r_g))
                    if val_g > best_val:
                        best_val = val_g
                        best_pos = p_g.copy()
                        best_rad = r_g.copy()

    # If we still have time, do extra random restarts from best
    if best_pos is not None and _remaining() > 25.0:
        for _ in range(10):
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
