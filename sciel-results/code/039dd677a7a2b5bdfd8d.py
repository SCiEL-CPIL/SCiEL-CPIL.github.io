"""Circle packing in unit square: maximize sum of radii of 26 disjoint circles in [0,1]^2."""
import time
import numpy as np
from scipy.optimize import minimize

N = 26
_DEADLINE = 145.0
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


def _maximize_radii(pos, rad):
    """
    For fixed positions, maximize each radius such that:
    1. Circle i fits in the box.
    2. Circle i does not overlap with any other circle j (assuming r_j is fixed at current value).
    Uses a more robust iterative scheme with damping to ensure convergence.
    """
    n = len(rad)
    pos = pos.copy()
    rad = rad.copy()
    
    # Precompute pairwise distances
    dx = pos[:, 0][:, None] - pos[:, 0][None, :]
    dy = pos[:, 1][:, None] - pos[:, 1][None, :]
    dist = np.sqrt(dx * dx + dy * dy)
    np.fill_diagonal(dist, np.inf)
    
    # Wall constraints for each circle
    wall_max = np.minimum(
        np.minimum(pos[:, 0], 1.0 - pos[:, 0]),
        np.minimum(pos[:, 1], 1.0 - pos[:, 1])
    )
    
    # Initialize radii to small positive values to avoid zero-division issues
    rad = np.maximum(rad, 1e-10)
    
    # Iterative relaxation with damping
    for iteration in range(100):
        changed = False
        for i in range(n):
            # Max radius from wall constraints
            max_r = wall_max[i]
            
            # Max radius from neighbor constraints: r_i <= d_ij - r_j for all j
            for j in range(n):
                if i != j:
                    limit = dist[i, j] - rad[j]
                    if limit < max_r:
                        max_r = limit
            
            max_r = max(0.0, max_r)
            
            # Damped update to ensure convergence
            if abs(max_r - rad[i]) > 1e-10:
                rad[i] = 0.5 * rad[i] + 0.5 * max_r
                changed = True
            else:
                rad[i] = max_r
        
        if not changed:
            break
    
    # Final exact pass without damping
    for i in range(n):
        max_r = wall_max[i]
        for j in range(n):
            if i != j:
                limit = dist[i, j] - rad[j]
                if limit < max_r:
                    max_r = limit
        rad[i] = max(0.0, max_r)
    
    return pos, rad


def _soft_objective(z, lam):
    """
    z = [x0,y0,r0, x1,y1,r1, ...]. 
    Minimize -sum(r) + lam*(overlap + wall penalties).
    Using sum of squares of individual violations for better conditioning.
    """
    n = N
    x = z[0::3]
    y = z[1::3]
    r = z[2::3]
    base = -np.sum(r)
    pen = 0.0
    
    # Wall penalties (smooth, sum of squares)
    # Violation is max(0, r - x), etc.
    v_left = np.maximum(r - x, 0.0)
    v_right = np.maximum(x + r - 1.0, 0.0)
    v_bot = np.maximum(r - y, 0.0)
    v_top = np.maximum(y + r - 1.0, 0.0)
    
    pen += np.sum(v_left ** 2) + np.sum(v_right ** 2) + np.sum(v_bot ** 2) + np.sum(v_top ** 2)
    
    # Pairwise overlap - vectorized
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d2 = dx * dx + dy * dy
    need = r[:, None] + r[None, :]
    
    iu = np.triu_indices(n, 1)
    d2u = d2[iu]
    needu = need[iu]
    
    # Distance
    du = np.sqrt(np.maximum(d2u, 1e-12))
    # Violation is max(0, need - dist)
    v_ov = np.maximum(needu - du, 0.0)
    pen += np.sum(v_ov ** 2)
    
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
    
    # Wall gradients
    v_left = np.maximum(r - x, 0.0)
    if np.any(v_left > 0):
        gx[v_left > 0] += lam * 2.0 * v_left[v_left > 0] * (-1.0)
        gr[v_left > 0] += lam * 2.0 * v_left[v_left > 0]
        
    v_right = np.maximum(x + r - 1.0, 0.0)
    if np.any(v_right > 0):
        gx[v_right > 0] += lam * 2.0 * v_right[v_right > 0] * (1.0)
        gr[v_right > 0] += lam * 2.0 * v_right[v_right > 0]
        
    v_bot = np.maximum(r - y, 0.0)
    if np.any(v_bot > 0):
        gy[v_bot > 0] += lam * 2.0 * v_bot[v_bot > 0] * (-1.0)
        gr[v_bot > 0] += lam * 2.0 * v_bot[v_bot > 0]
        
    v_top = np.maximum(y + r - 1.0, 0.0)
    if np.any(v_top > 0):
        gy[v_top > 0] += lam * 2.0 * v_top[v_top > 0] * (1.0)
        gr[v_top > 0] += lam * 2.0 * v_top[v_top > 0]
        
    # Pairwise gradients
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d2 = dx * dx + dy * dy
    need = r[:, None] + r[None, :]
    
    iu = np.triu_indices(n, 1)
    d2u = d2[iu]
    needu = need[iu]
    
    du = np.sqrt(np.maximum(d2u, 1e-12))
    v_ov = np.maximum(needu - du, 0.0)
    
    mask = v_ov > 0
    if np.any(mask):
        idx_i = iu[0][mask]
        idx_j = iu[1][mask]
        dxu = dx[iu][mask]
        dyu = dy[iu][mask]
        du_m = du[mask]
        v_m = v_ov[mask]
        
        inv_u = 1.0 / du_m
        # d(v^2)/dx_i = 2*v * (-dx/d)
        np.add.at(gx, idx_i, lam * 2.0 * v_m * (-dxu * inv_u))
        np.add.at(gx, idx_j, lam * 2.0 * v_m * (dxu * inv_u))
        np.add.at(gy, idx_i, lam * 2.0 * v_m * (-dyu * inv_u))
        np.add.at(gy, idx_j, lam * 2.0 * v_m * (dyu * inv_u))
        # d(v^2)/dr_i = 2*v * (1)
        np.add.at(gr, idx_i, lam * 2.0 * v_m)
        np.add.at(gr, idx_j, lam * 2.0 * v_m)
        
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

    def _update_best(p, r):
        nonlocal best_val, best_pos, best_rad
        # Ensure feasibility
        val, feas = _feasible_score(p, r)
        if feas:
            if val > best_val:
                best_val = val
                best_pos = p.copy()
                best_rad = r.copy()
                return True
        return False

    # Anneal lambda: start high (strong penalty) then lower to grow radii
    # The "tighten" step (_maximize_radii) is the key improvement.
    # We run optimization to find good positions, then tighten radii.
    
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
            
            # KEY STEP: Maximize radii for these positions
            cur_pos, cur_rad = _maximize_radii(cur_pos, cur_rad)
            
            _update_best(cur_pos, cur_rad)

    # If we still have time, do extra random restarts from best
    if best_pos is not None and _remaining() > 30.0:
        for _ in range(15):
            if _remaining() < 10.0:
                break
            p = best_pos + rng.normal(0, 0.03, size=best_pos.shape)
            p = np.clip(p, 0.05, 0.95)
            # Start with small radii to allow optimizer to find space, or use current best scaled down
            r = best_rad * 0.8 
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
                # KEY STEP: Maximize radii
                cur_pos, cur_rad = _maximize_radii(cur_pos, cur_rad)
                _update_best(cur_pos, cur_rad)

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
            # KEY STEP: Maximize radii
            p, r = _maximize_radii(p, r)
            _update_best(p, r)
        
        # Final radius growth polish (though _maximize_radii should have handled it, 
        # a final check ensures no numerical drift allowed shrinkage)
        if _remaining() > 5.0:
            p, r = _maximize_radii(best_pos, best_rad)
            _update_best(p, r)

    # Fallback: guaranteed feasible seed if nothing found
    if best_pos is None:
        p, r = _hex_seed(jitter=0.0)
        p, r = _maximize_radii(p, r)
        _update_best(p, r)
        
    if best_pos is None:
        # absolute fallback: tiny circles in a grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        arr = np.array(cells[:N])
        return arr

    out = np.column_stack([best_pos[:, 0], best_pos[:, 1], best_rad])
    return out


if __name__ == '__main__':
    res = construct()
    print("shape", res.shape)
    print("sum r", res[:, 2].sum())
    print("feasible", _feasible_score(res[:, :2], res[:, 2]))
