"""Circle packing: Voronoi inradius-based optimization with multi-start."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Voronoi, Delaunay

N = 26
_START = time.time()
_DEADLINE = 150.0
_EPS = 1e-9


def _voronoi_inradii(points):
    """Compute the inradius (max radius of circle centered at point[i] 
    that fits within the Voronoi cell and the unit square) for each point."""
    n = len(points)
    inradii = np.zeros(n)
    
    # Create Voronoi diagram
    try:
        vor = Voronoi(points)
    except Exception:
        # Fallback: use distance to nearest other point / 2
        for i in range(n):
            dmin = min(np.sqrt((points[i] - points[j])**2).sum() for j in range(n) if j != i)
            inradii[i] = max(dmin * 0.5, 1e-4)
        return inradii
    
    # For each point, find its Voronoi cell and compute inradius
    for i in range(n):
        # Get the Voronoi region (list of vertex indices)
        region_idx = vor.regions[vor.point_region[i]]
        if len(region_idx) == 0:
            inradii[i] = 1e-4
            continue
        
        vertices = vor.vertices[region_idx]
        if len(vertices) == 0:
            inradii[i] = 1e-4
            continue
        
        # The inradius is the minimum distance from point[i] to any edge of the cell
        # Edges are between consecutive vertices (and the region may be unbounded)
        min_dist = 0.5  # max possible radius in unit square
        
        # Distance to unit square boundaries
        px, py = points[i]
        min_dist = min(min_dist, px, py, 1.0 - px, 1.0 - py)
        
        # Distance to each edge of the Voronoi cell
        # Voronoi vertices form a polygon; edges are between consecutive vertices
        # But we need to be careful: the region is defined by half-planes
        # Better approach: distance to each edge defined by consecutive vertices
        nv = len(vertices)
        for k in range(nv):
            v1 = vertices[k]
            v2 = vertices[(k + 1) % nv]
            # Check if this edge is finite (both vertices exist)
            if vor.vertices[v1] is None or vor.vertices[v2] is None:
                continue
            # Distance from point to line through v1, v2
            x1, y1 = vor.vertices[v1]
            x2, y2 = vor.vertices[v2]
            dx = x2 - x1
            dy = y2 - y1
            length = np.sqrt(dx*dx + dy*dy)
            if length < 1e-12:
                continue
            # Distance from point to line
            dist = abs(dy*(px - x1) - dx*(py - y1)) / length
            min_dist = min(min_dist, dist)
        
        inradii[i] = max(min_dist, 1e-4)
    
    return inradii


def _voronoi_objective(p):
    """Negative sum of Voronoi inradii (to be minimized)."""
    n = N
    points = p.reshape(n, 2)
    inradii = _voronoi_inradii(points)
    return -np.sum(inradii)


def _valid_and_score(arr):
    """Return sum of radii if valid (disjoint, in square), else -inf."""
    x = arr[:, 0]
    y = arr[:, 1]
    r = arr[:, 2]
    if np.any(r <= 0) or np.any(np.isnan(arr)):
        return -np.inf
    if np.any(x < -_EPS) or np.any(x > 1 + _EPS) or np.any(y < -_EPS) or np.any(y > 1 + _EPS):
        return -np.inf
    if np.any(r < -_EPS) or np.any(x + r > 1 + _EPS) or np.any(x - r < -_EPS) \
            or np.any(y + r > 1 + _EPS) or np.any(y - r < -_EPS):
        return -np.inf
    for i in range(N):
        for j in range(i + 1, N):
            d = np.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2)
            if d < r[i] + r[j] - 1e-7:
                return -np.inf
    return float(np.sum(r))


def _smooth_objective(p):
    """Penalty-based objective for smooth optimization (positions + radii)."""
    n = len(p)
    x = p[:2 * n]
    r = p[2 * n:]
    val = np.sum(r)
    b = np.maximum(0.0, r - x) + np.maximum(0.0, r - (1.0 - x))
    val -= 100.0 * np.sum(b)
    for i in range(n):
        dx = x[i] - x[i + 1:n]
        dy = x[i + n:] - x[i + n + 1:n + (n - i - 1)]
        d = np.sqrt(dx * dx + dy * dy)
        s = r[i] + r[i + 1:n]
        pen = np.maximum(0.0, s - d)
        val -= 100.0 * np.sum(pen)
    return val


def _polish(p):
    """Hard-constraint local optimization: maximize sum r subject to constraints."""
    n = len(p)
    x = p[:2 * n].copy()
    r = p[2 * n:].copy()

    def f(v):
        return -np.sum(v[2 * n:])

    cons = []
    for i in range(n):
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[i] - v[2 * n + i]})
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[i + n] - v[2 * n + i]})
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: 1.0 - v[i] - v[2 * n + i]})
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: 1.0 - v[i + n] - v[2 * n + i]})
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({
                'type': 'ineq',
                'fun': lambda v, i=i, j=j:
                    np.sqrt((v[i] - v[j]) ** 2 + (v[i + n] - v[j + n]) ** 2)
                    - v[2 * n + i] - v[2 * n + j],
            })
    bounds = [(0.0, 1.0)] * (2 * n) + [(1e-4, 0.5)] * n
    res = minimize(f, p, method='SLSQP', bounds=bounds, constraints=cons,
                   options={'maxiter': 200, 'ftol': 1e-9})
    if res.success or (hasattr(res, 'fun') and res.fun < f(p)):
        return res.x
    return p


def construct():
    n = N
    best_arr = None
    best_val = -np.inf
    rng = np.random.default_rng(12345)
    
    def _update_best(arr):
        nonlocal best_arr, best_val
        val = _valid_and_score(arr)
        if val > best_val:
            best_val = val
            best_arr = arr.copy()
    
    # ---- Phase 1: Voronoi inradius optimization (positions only) ----
    n_starts_voro = 0
    max_starts_voro = 30
    while time.time() < _DEADLINE - 40.0 and n_starts_voro < max_starts_voro:
        # Random initial positions
        x = rng.uniform(0.05, 0.95, size=n)
        y = rng.uniform(0.05, 0.95, size=n)
        p0 = np.concatenate([x, y])
        
        res = minimize(_voronoi_objective, p0, method='L-BFGS-B',
                       options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8})
        p = res.x
        x_opt = p[:n]
        y_opt = p[n:2*n]
        points = np.column_stack([x_opt, y_opt])
        
        # Compute inradii from Voronoi
        inradii = _voronoi_inradii(points)
        arr = np.column_stack([x_opt, y_opt, inradii])
        arr[:, 0] = np.clip(arr[:, 0], 0, 1)
        arr[:, 1] = np.clip(arr[:, 1], 0, 1)
        arr[:, 2] = np.clip(arr[:, 2], 1e-4, 0.5)
        
        _update_best(arr)
        n_starts_voro += 1
    
    # ---- Phase 2: Smooth penalty-based multi-start (positions + radii) ----
    n_starts_smooth = 0
    max_starts_smooth = 20
    while time.time() < _DEADLINE - 20.0 and n_starts_smooth < max_starts_smooth:
        x = rng.uniform(0.05, 0.95, size=n)
        y = rng.uniform(0.05, 0.95, size=n)
        r = rng.uniform(0.02, 0.15, size=n)
        p0 = np.concatenate([x, y, r])
        res = minimize(_smooth_objective, p0, method='L-BFGS-B',
                       options={'maxiter': 150, 'ftol': 1e-10, 'gtol': 1e-8})
        p = res.x
        arr = np.column_stack([p[:n], p[n:2*n], p[2*n:]])
        arr[:, 0] = np.clip(arr[:, 0], 0, 1)
        arr[:, 1] = np.clip(arr[:, 1], 0, 1)
        arr[:, 2] = np.clip(arr[:, 2], 1e-4, 0.5)
        _update_best(arr)
        n_starts_smooth += 1
    
    # ---- Phase 3: Polish the best candidate with hard constraints ----
    if best_arr is not None and time.time() < _DEADLINE - 5.0:
        p = np.concatenate([best_arr[:, 0], best_arr[:, 1], best_arr[:, 2]])
        pp = _polish(p)
        arr = np.column_stack([pp[:n], pp[n:2*n], pp[2*n:]])
        arr[:, 0] = np.clip(arr[:, 0], 0, 1)
        arr[:, 1] = np.clip(arr[:, 1], 0, 1)
        arr[:, 2] = np.clip(arr[:, 2], 1e-4, 0.5)
        _update_best(arr)
    
    # ---- Fallback: ensure we return something valid ----
    if best_arr is None or not np.isfinite(best_val):
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        best_arr = np.array(cells[:N])
    
    # final safety clip
    best_arr[:, 0] = np.clip(best_arr[:, 0], 0, 1)
    best_arr[:, 1] = np.clip(best_arr[:, 1], 0, 1)
    best_arr[:, 2] = np.clip(best_arr[:, 2], 1e-4, 0.5)
    return best_arr
