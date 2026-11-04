"""Circle packing in unit square: maximize sum of radii of 26 disjoint circles in [0,1]^2."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import Voronoi, QhullError

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


def _voronoi_inradii(pos):
    """
    Compute the inradius of each Voronoi cell for points in [0,1]^2.
    Each circle's radius is limited by its Voronoi cell's inradius.
    Returns array of inradii (one per point).
    """
    n = len(pos)
    # Add boundary points to make Voronoi diagram bounded
    # Use a large square boundary
    margin = 2.0
    boundary_pts = np.array([
        [-margin, -margin], [1 + margin, -margin],
        [1 + margin, 1 + margin], [-margin, 1 + margin],
        [-margin, 0.5], [1 + margin, 0.5],
        [0.5, -margin], [0.5, 1 + margin]
    ])
    all_pts = np.vstack([pos, boundary_pts])
    
    inradii = np.zeros(n)
    
    try:
        vor = Voronoi(all_pts)
    except QhullError:
        # Fallback: use min distance to other points and walls
        for i in range(n):
            dmin = min(pos[i, 0], 1 - pos[i, 0], pos[i, 1], 1 - pos[i, 1])
            for j in range(n):
                if i != j:
                    dmin = min(dmin, np.hypot(pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1]))
            inradii[i] = 0.5 * dmin
        return inradii
    
    # For each original point, compute inradius of its Voronoi cell
    for i in range(n):
        # Get the Voronoi region for point i
        region_idx = vor.point_region[i]
        region = vor.regions[region_idx]
        
        if len(region) == 0 or -1 in region:
            # Unbounded region (shouldn't happen with our boundary points, but just in case)
            dmin = min(pos[i, 0], 1 - pos[i, 0], pos[i, 1], 1 - pos[i, 1])
            for j in range(n):
                if i != j:
                    dmin = min(dmin, np.hypot(pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1]))
            inradii[i] = 0.5 * dmin
            continue
        
        # Compute inradius: minimum distance from point i to any edge of its cell
        min_dist = np.inf
        for edge_idx in region:
            edge = vor.ridges[edge_idx]
            # edge is [vertex1, vertex2]
            v1 = vor.vertices[edge[0]]
            v2 = vor.vertices[edge[1]]
            
            # Distance from point i to line segment v1-v2
            # Project point onto line
            d_vec = v2 - v1
            d_len_sq = np.dot(d_vec, d_vec)
            if d_len_sq < 1e-15:
                dist = np.hypot(pos[i, 0] - v1[0], pos[i, 1] - v1[1])
            else:
                t = np.dot(pos[i] - v1, d_vec) / d_len_sq
                t = np.clip(t, 0, 1)
                proj = v1 + t * d_vec
                dist = np.hypot(pos[i, 0] - proj[0], pos[i, 1] - proj[1])
            
            min_dist = min(min_dist, dist)
        
        # Also check distance to walls
        wall_dist = min(pos[i, 0], 1 - pos[i, 0], pos[i, 1], 1 - pos[i, 1])
        inradii[i] = min(min_dist, wall_dist)
    
    return inradii


def _voronoi_objective(pos_flat):
    """
    Objective for Voronoi-based optimization.
    pos_flat: 2D array of shape (n, 2) flattened to 1D.
    Minimize negative sum of Voronoi inradii.
    """
    pos = pos_flat.reshape(-1, 2)
    # Clamp to valid range
    pos = np.clip(pos, 0.01, 0.99)
    inradii = _voronoi_inradii(pos)
    return -np.sum(inradii)


def _voronoi_gradient(pos_flat):
    """
    Numerical gradient for Voronoi-based optimization.
    """
    n = len(pos_flat) // 2
    pos = pos_flat.reshape(-1, 2)
    pos = np.clip(pos, 0.01, 0.99)
    
    grad = np.zeros_like(pos_flat)
    h = 1e-5
    
    for k in range(len(pos_flat)):
        pos_plus = pos.copy()
        pos_plus.flat[k] += h
        pos_plus = np.clip(pos_plus, 0.01, 0.99)
        
        pos_minus = pos.copy()
        pos_minus.flat[k] -= h
        pos_minus = np.clip(pos_minus, 0.01, 0.99)
        
        f_plus = _voronoi_inradii(pos_plus)
        f_minus = _voronoi_inradii(pos_minus)
        
        grad[k] = (np.sum(f_plus) - np.sum(f_minus)) / (2 * h)
    
    return grad


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
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d2 = dx * dx + dy * dy
    need = r[:, None] + r[None, :]
    iu = np.triu_indices(n, 1)
    d2u = d2[iu]
    needu = need[iu]
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
    gr -= 1.0
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
        fu = needu[mask] - du
        inv_u = 1.0 / du
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


def _grow_radii_voronoi(pos, rng, max_grow=200, growth_step=0.001):
    """
    Grow radii using Voronoi inradii. At each step, compute inradii and
    use them as the radii. If infeasible, shrink and adjust positions.
    Returns (pos, rad, feasible).
    """
    pos = pos.copy()
    n = len(pos)
    
    for _ in range(max_grow):
        if _remaining() < 5.0:
            break
        
        # Compute Voronoi inradii
        rad = _voronoi_inradii(pos)
        
        # Check feasibility
        val, feas = _feasible_score(pos, rad)
        if feas:
            return pos, rad, True
        
        # If not feasible, shrink radii and try to adjust positions
        rad *= 0.95
        pos, rad, feas = _project_feasible(pos, rad)
        if feas:
            return pos, rad, True
    
    # Final attempt
    rad = _voronoi_inradii(pos)
    pos, rad, feas = _project_feasible(pos, rad)
    return pos, rad, feas


def construct():
    global _START
    _START = time.time()
    rng = np.random.default_rng(12345)

    best_val = -np.inf
    best_pos = None
    best_rad = None

    # Generate initial seeds
    seeds = []
    for jitter in [0.0, 0.02, 0.05, 0.1]:
        p, r = _hex_seed(jitter=jitter, rng=rng)
        seeds.append(p)
    for _ in range(6):
        p = rng.uniform(0.15, 0.85, size=(N, 2))
        seeds.append(p)

    # Phase 1: Voronoi-based optimization
    # For each seed, optimize positions to maximize sum of Voronoi inradii
    for si, seed_pos in enumerate(seeds):
        if _remaining() < 20.0:
            break
        
        # Try multiple optimization approaches
        # Approach 1: L-BFGS-B with numerical gradient
        pos_flat = seed_pos.flatten()
        bounds_v = [(0.01, 0.99)] * (2 * N)
        
        try:
            res = minimize(_voronoi_objective, pos_flat, method='L-BFGS-B',
                           bounds=bounds_v,
                           options={'maxiter': 200, 'gtol': 1e-6, 'ftol': 1e-8})
            opt_pos = res.x.reshape(-1, 2)
            
            # Compute inradii and check feasibility
            rad = _voronoi_inradii(opt_pos)
            opt_pos, rad, feas = _project_feasible(opt_pos, rad)
            if feas:
                val = float(np.sum(rad))
                if val > best_val:
                    best_val = val
                    best_pos = opt_pos.copy()
                    best_rad = rad.copy()
        except Exception:
            pass
        
        # Approach 2: Differential evolution (global search)
        if _remaining() > 30.0:
            try:
                de_res = differential_evolution(_voronoi_objective, bounds_v,
                                                seed=int(rng.integers(0, 2**31)),
                                                maxiter=50, popsize=15,
                                                tol=1e-6, init='sobol')
                de_pos = de_res.x.reshape(-1, 2)
                rad = _voronoi_inradii(de_pos)
                de_pos, rad, feas = _project_feasible(de_pos, rad)
                if feas:
                    val = float(np.sum(rad))
                    if val > best_val:
                        best_val = val
                        best_pos = de_pos.copy()
                        best_rad = rad.copy()
            except Exception:
                pass
    
    # Phase 2: Refine best solution with local optimization
    if best_pos is not None and _remaining() > 25.0:
        # Multiple local refinements with perturbations
        for trial in range(5):
            if _remaining() < 10.0:
                break
            
            # Perturb the best solution
            perturbation = 0.02 * (1 - trial * 0.15)
            p = best_pos + rng.normal(0, perturbation, size=best_pos.shape)
            p = np.clip(p, 0.05, 0.95)
            
            # Optimize from perturbed position
            pos_flat = p.flatten()
            bounds_v = [(0.01, 0.99)] * (2 * N)
            
            try:
                res = minimize(_voronoi_objective, pos_flat, method='L-BFGS-B',
                               bounds=bounds_v,
                               options={'maxiter': 300, 'gtol': 1e-8, 'ftol': 1e-10})
                opt_pos = res.x.reshape(-1, 2)
                rad = _voronoi_inradii(opt_pos)
                opt_pos, rad, feas = _project_feasible(opt_pos, rad)
                if feas:
                    val = float(np.sum(rad))
                    if val > best_val:
                        best_val = val
                        best_pos = opt_pos.copy()
                        best_rad = rad.copy()
            except Exception:
                pass
    
    # Phase 3: High-precision final polish
    if best_pos is not None and _remaining() > 15.0:
        # Very tight optimization
        pos_flat = best_pos.flatten()
        bounds_v = [(0.01, 0.99)] * (2 * N)
        
        try:
            res = minimize(_voronoi_objective, pos_flat, method='L-BFGS-B',
                           bounds=bounds_v,
                           options={'maxiter': 500, 'gtol': 1e-10, 'ftol': 1e-12})
            opt_pos = res.x.reshape(-1, 2)
            rad = _voronoi_inradii(opt_pos)
            opt_pos, rad, feas = _project_feasible(opt_pos, rad, iters=60)
            if feas:
                val = float(np.sum(rad))
                if val > best_val:
                    best_val = val
                    best_pos = opt_pos.copy()
                    best_rad = rad.copy()
        except Exception:
            pass
    
    # Fallback: if nothing found, use hex seed
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
