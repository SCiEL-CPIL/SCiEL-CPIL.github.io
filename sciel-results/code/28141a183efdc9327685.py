"""Circle packing in unit square: 26 disjoint circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay, Voronoi
from scipy.spatial.distance import cdist

N = 26
EPS = 1e-8
DEADLINE = 145.0  # safety margin before 150s kill


def _pairwise_constraints(p, iu):
    """Return pairwise distance - (r_i + r_j) for all pairs."""
    x = p[:, 0]
    y = p[:, 1]
    r = np.maximum(p[:, 2], 1e-6)
    dx = x[iu[0]] - x[iu[1]]
    dy = y[iu[0]] - y[iu[1]]
    dist = np.sqrt(dx * dx + dy * dy)
    return dist - (r[iu[0]] + r[iu[1]])


def _pairwise_grad(p, iu):
    """Gradient of pairwise constraints."""
    x = p[:, 0]
    y = p[:, 1]
    r = np.maximum(p[:, 2], 1e-6)
    n = N
    g = np.zeros((len(iu[0]), 3 * N))
    
    dx = x[iu[0]] - x[iu[1]]
    dy = y[iu[0]] - y[iu[1]]
    dist = np.sqrt(dx * dx + dy * dy)
    dist_safe = np.maximum(dist, 1e-10)
    
    ux = dx / dist_safe
    uy = dy / dist_safe
    
    for k in range(len(iu[0])):
        i = iu[0][k]
        j = iu[1][k]
        g[k, 3*i] = ux[k]
        g[k, 3*i+1] = uy[k]
        g[k, 3*i+2] = -1.0
        g[k, 3*j] = -ux[k]
        g[k, 3*j+1] = -uy[k]
        g[k, 3*j+2] = -1.0
    
    return g


def _wall_constraints(p):
    """Return wall clearance: min(x-r, 1-x-r, y-r, 1-y-r) per circle."""
    x = p[:, 0]
    y = p[:, 1]
    r = np.maximum(p[:, 2], 1e-6)
    return np.minimum(np.minimum(x - r, 1.0 - x - r),
                      np.minimum(y - r, 1.0 - y - r))


def _wall_grad(p):
    """Gradient of wall constraints."""
    x = p[:, 0]
    y = p[:, 1]
    r = np.maximum(p[:, 2], 1e-6)
    n = N
    g = np.zeros((n, 3 * N))
    
    for i in range(n):
        wx_left = x[i] - r[i]
        wx_right = 1.0 - x[i] - r[i]
        wy_bottom = y[i] - r[i]
        wy_top = 1.0 - y[i] - r[i]
        
        wmin = min(wx_left, wx_right, wy_bottom, wy_top)
        
        if wx_left == wmin:
            g[i, 3*i] = 1.0
            g[i, 3*i+2] = -1.0
        if wx_right == wmin:
            g[i, 3*i] = -1.0
            g[i, 3*i+2] = -1.0
        if wy_bottom == wmin:
            g[i, 3*i+1] = 1.0
            g[i, 3*i+2] = -1.0
        if wy_top == wmin:
            g[i, 3*i+1] = -1.0
            g[i, 3*i+2] = -1.0
    
    return g


def _penalty_objective(params, penalty=1000.0):
    """Penalty objective: minimize -sum(r) + penalty*(overlaps + wall violations)."""
    p = params.reshape(N, 3)
    x = p[:, 0]
    y = p[:, 1]
    r = np.maximum(p[:, 2], 1e-6)

    score = -np.sum(r)

    # Wall boundary violations
    wx = np.maximum(0.0, r - x) + np.maximum(0.0, r - (1.0 - x))
    wy = np.maximum(0.0, r - y) + np.maximum(0.0, r - (1.0 - y))
    wall = np.sum(wx) + np.sum(wy)

    # Pairwise overlap violations
    iu = np.triu_indices(N, k=1)
    dx = x[iu[0]] - x[iu[1]]
    dy = y[iu[0]] - y[iu[1]]
    dist = np.sqrt(dx * dx + dy * dy)
    overlap = np.maximum(0.0, (r[iu[0]] + r[iu[1]]) - dist)
    pair = np.sum(overlap)

    return score + penalty * (pair + wall)


def _penalty_grad(params, penalty=1000.0):
    """Analytic gradient of penalty objective."""
    p = params.reshape(N, 3)
    x = p[:, 0]
    y = p[:, 1]
    r = np.maximum(p[:, 2], 1e-6)
    n = N
    g = np.zeros_like(params)

    # d(-sum r)/dr = -1
    g[2::3] = -1.0

    # Wall contributions
    for i in range(n):
        if r[i] - x[i] > 0:
            g[3*i] += penalty * (-1.0)
            g[3*i+2] += penalty * (+1.0)
        if r[i] - (1.0 - x[i]) > 0:
            g[3*i] += penalty * (+1.0)
            g[3*i+2] += penalty * (+1.0)
        if r[i] - y[i] > 0:
            g[3*i+1] += penalty * (-1.0)
            g[3*i+2] += penalty * (+1.0)
        if r[i] - (1.0 - y[i]) > 0:
            g[3*i+1] += penalty * (+1.0)
            g[3*i+2] += penalty * (+1.0)

    # Pairwise
    iu = np.triu_indices(n, k=1)
    dx = x[iu[0]] - x[iu[1]]
    dy = y[iu[0]] - y[iu[1]]
    dist = np.sqrt(dx * dx + dy * dy)
    dist_safe = np.maximum(dist, 1e-10)
    ov = (r[iu[0]] + r[iu[1]]) - dist
    active = ov > 0
    if np.any(active):
        ax = iu[0][active]
        ay = iu[1][active]
        adx = dx[active] / dist_safe[active]
        ady = dy[active] / dist_safe[active]
        for k in range(len(ax)):
            i = ax[k]
            j = ay[k]
            g[3*i] += penalty * (-adx[k])
            g[3*j] += penalty * (+adx[k])
            g[3*i+1] += penalty * (-ady[k])
            g[3*j+1] += penalty * (+ady[k])
            g[3*i+2] += penalty * (+1.0)
            g[3*j+2] += penalty * (+1.0)

    return g


def _hex_initial(seed):
    """Hexagonal grid initialization."""
    rng = np.random.RandomState(seed)
    points = []
    for i in range(7):
        for j in range(7):
            x = i
            y = j * 1.5 + (0.5 if i % 2 else 0)
            points.append((x, y))
    
    points = np.array(points)
    points -= np.mean(points, axis=0)
    max_coord = np.max(np.abs(points))
    scale = 0.45 / max_coord
    points *= scale
    points += 0.5
    
    dists = np.sum((points - 0.5)**2, axis=1)
    indices = np.argsort(dists)[:N]
    
    p = points[indices]
    radii = np.full(N, 0.05)
    
    params = np.empty(3 * N)
    params[0::3] = p[:, 0]
    params[1::3] = p[:, 1]
    params[2::3] = radii
    
    params += rng.uniform(-1e-4, 1e-4, size=params.size)
    return params


def _grid_initial(seed):
    """Grid of equal circles with small jitter."""
    rng = np.random.RandomState(seed)
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            cells.append(((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-4))
    cells = cells[:N]
    arr = np.array(cells, dtype=float).reshape(-1)
    arr += rng.uniform(-1e-5, 1e-5, size=arr.size)
    return arr


def _voronoi_initial(seed):
    """Initialize using Voronoi cells to estimate max possible radii."""
    rng = np.random.RandomState(seed)
    points = rng.uniform(0.05, 0.95, size=(N, 2))
    
    radii = np.zeros(N)
    for i, p in enumerate(points):
        dists = np.linalg.norm(points - p, axis=1)
        dists[i] = np.inf
        min_dist_neighbor = np.min(dists)
        wall_dist = np.min([p[0], 1-p[0], p[1], 1-p[1]])
        radii[i] = 0.5 * np.min([min_dist_neighbor, wall_dist]) * 0.95
        
    params = np.empty(3 * N)
    params[0::3] = points[:, 0]
    params[1::3] = points[:, 1]
    params[2::3] = radii
    params += rng.uniform(-1e-5, 1e-5, size=params.size)
    return params


def _bounds():
    lo = np.zeros(3 * N)
    hi = np.zeros(3 * N)
    for i in range(N):
        lo[3*i] = 0.0
        hi[3*i] = 1.0
        lo[3*i+1] = 0.0
        hi[3*i+1] = 1.0
        lo[3*i+2] = 1e-6
        hi[3*i+2] = 0.5
    return (lo, hi)


def _enforce_validity(params):
    """Enforce all constraints: walls and pairwise non-overlap by shrinking radii."""
    p = params.reshape(N, 3).copy()
    p[:, 0] = np.clip(p[:, 0], 0.0, 1.0)
    p[:, 1] = np.clip(p[:, 1], 0.0, 1.0)
    p[:, 2] = np.clip(p[:, 2], 1e-6, 0.5)

    wall = np.minimum(np.minimum(p[:, 0], 1.0 - p[:, 0]),
                      np.minimum(p[:, 1], 1.0 - p[:, 1]))
    p[:, 2] = np.minimum(p[:, 2], np.maximum(wall, 1e-6))

    for _ in range(1000):
        iu = np.triu_indices(N, k=1)
        dx = p[iu[0], 0] - p[iu[1], 0]
        dy = p[iu[0], 1] - p[iu[1], 1]
        dist = np.sqrt(dx * dx + dy * dy)
        overlap = (p[iu[0], 2] + p[iu[1], 2]) - dist
        active = overlap > 0
        if not np.any(active):
            break
        for k in range(len(active)):
            if not active[k]:
                continue
            i = iu[0][k]
            j = iu[1][k]
            ov = overlap[k]
            if p[i, 2] <= p[j, 2]:
                shrink = min(ov, p[i, 2] - 1e-6)
                p[i, 2] -= shrink
                rem = ov - shrink
                if rem > 0:
                    shrink2 = min(rem, p[j, 2] - 1e-6)
                    p[j, 2] -= shrink2
            else:
                shrink = min(ov, p[j, 2] - 1e-6)
                p[j, 2] -= shrink
                rem = ov - shrink
                if rem > 0:
                    shrink2 = min(rem, p[i, 2] - 1e-6)
                    p[i, 2] -= shrink2
        wall = np.minimum(np.minimum(p[:, 0], 1.0 - p[:, 0]),
                          np.minimum(p[:, 1], 1.0 - p[:, 1]))
        p[:, 2] = np.minimum(p[:, 2], np.maximum(wall, 1e-6))

    return p


def _score_valid(p):
    """Compute sum of radii for a valid configuration."""
    return np.sum(p[:, 2])


def _delaunay_refine(params, deadline, bounds_list):
    """
    Structural refinement using Delaunay triangulation.
    Only optimize over edges in the Delaunay graph, reducing constraint count
    and allowing SLSQP to work on a sparse, geometrically meaningful structure.
    """
    p = params.reshape(N, 3).copy()
    centers = p[:, :2]
    
    # Build Delaunay triangulation
    try:
        tri = Delaunay(centers)
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i+1, 3):
                    a, b = simplex[i], simplex[j]
                    if a > b:
                        a, b = b, a
                    edges.add((a, b))
    except Exception:
        # Fallback: use all pairs
        iu = np.triu_indices(N, k=1)
        edges = set(zip(iu[0], iu[1]))
    
    edge_list = list(edges)
    if len(edge_list) == 0:
        return params
    
    # Create constraint arrays for SLSQP
    eu = np.array([e[0] for e in edge_list])
    ev = np.array([e[1] for e in edge_list])
    
    def obj(x):
        """Maximize sum of radii (minimize negative)."""
        return -np.sum(x[2::3])
    
    def obj_grad(x):
        g = np.zeros_like(x)
        g[2::3] = -1.0
        return g
    
    def constraint_func(x):
        """Delaunay edge distance constraints: dist(i,j) >= r_i + r_j."""
        xi = x[eu * 3]
        yi = x[eu * 3 + 1]
        ri = np.maximum(x[eu * 3 + 2], 1e-6)
        xj = x[ev * 3]
        yj = x[ev * 3 + 1]
        rj = np.maximum(x[ev * 3 + 2], 1e-6)
        dx = xi - xj
        dy = yi - yj
        dist = np.sqrt(dx * dx + dy * dy)
        return dist - (ri + rj)
    
    def constraint_grad(x):
        """Jacobian of Delaunay edge constraints."""
        m = len(eu)
        jacobian = np.zeros((m, 3 * N))
        xi = x[eu * 3]
        yi = x[eu * 3 + 1]
        ri = np.maximum(x[eu * 3 + 2], 1e-6)
        xj = x[ev * 3]
        yj = x[ev * 3 + 1]
        rj = np.maximum(x[ev * 3 + 2], 1e-6)
        dx = xi - xj
        dy = yi - yj
        dist = np.sqrt(dx * dx + dy * dy)
        dist_safe = np.maximum(dist, 1e-10)
        ux = dx / dist_safe
        uy = dy / dist_safe
        
        for k in range(m):
            i = eu[k]
            j = ev[k]
            jacobian[k, 3*i] = ux[k]
            jacobian[k, 3*i+1] = uy[k]
            jacobian[k, 3*i+2] = -1.0
            jacobian[k, 3*j] = -ux[k]
            jacobian[k, 3*j+1] = -uy[k]
            jacobian[k, 3*j+2] = -1.0
        return jacobian
    
    # Wall constraints for SLSQP
    def wall_func(x):
        """Wall clearance constraints."""
        xi = x[0::3]
        yi = x[1::3]
        ri = np.maximum(x[2::3], 1e-6)
        return np.minimum(np.minimum(xi - ri, 1.0 - xi - ri),
                          np.minimum(yi - ri, 1.0 - yi - ri))
    
    def wall_grad(x):
        """Jacobian of wall constraints."""
        m = N
        jacobian = np.zeros((m, 3 * N))
        xi = x[0::3]
        yi = x[1::3]
        ri = np.maximum(x[2::3], 1e-6)
        
        for i in range(m):
            wx_left = xi[i] - ri[i]
            wx_right = 1.0 - xi[i] - ri[i]
            wy_bottom = yi[i] - ri[i]
            wy_top = 1.0 - yi[i] - ri[i]
            wmin = min(wx_left, wx_right, wy_bottom, wy_top)
            
            if wx_left == wmin:
                jacobian[i, 3*i] = 1.0
                jacobian[i, 3*i+2] = -1.0
            if wx_right == wmin:
                jacobian[i, 3*i] = -1.0
                jacobian[i, 3*i+2] = -1.0
            if wy_bottom == wmin:
                jacobian[i, 3*i+1] = 1.0
                jacobian[i, 3*i+2] = -1.0
            if wy_top == wmin:
                jacobian[i, 3*i+1] = -1.0
                jacobian[i, 3*i+2] = -1.0
        return jacobian
    
    constraints = [
        {'type': 'ineq', 'fun': constraint_func, 'jac': constraint_grad},
        {'type': 'ineq', 'fun': wall_func, 'jac': wall_grad}
    ]
    
    try:
        res = minimize(obj, params, jac=obj_grad, method='SLSQP',
                       bounds=bounds_list, constraints=constraints,
                       options={'maxiter': 200, 'ftol': 1e-12, 'maxfun': 500})
        if res.success or _score_valid(_enforce_validity(res.x)) > _score_valid(_enforce_validity(params)):
            return res.x
    except Exception:
        pass
    
    return params


def _shake_and_refine(p, steps=5000, time_limit=20.0, seed=42):
    """
    Refine a valid configuration using random perturbations (shake and solve).
    This helps escape local minima found by gradient-based methods.
    """
    current_score = _score_valid(p)
    best_p = p.copy()
    best_score = current_score
    rng = np.random.RandomState(seed)
    
    start = time.time()
    sigma = 0.02
    
    for _ in range(steps):
        if time.time() - start > time_limit:
            break
            
        # Perturb positions and radii
        p_new = best_p.copy()
        p_new[:, :2] += rng.normal(0, sigma, size=(N, 2))
        p_new[:, 2] += rng.normal(0, sigma * 0.5, size=N)
        
        # Enforce validity
        p_new = _enforce_validity(p_new)
        new_score = _score_valid(p_new)
        
        if new_score > best_score:
            best_score = new_score
            best_p = p_new.copy()
        elif new_score > current_score - 0.01:
            # Accept worse solutions with some probability to escape local minima
            if rng.random() < 0.3:
                current_score = new_score
                best_p = p_new.copy()
        else:
            current_score = new_score
            best_p = p_new.copy()
    
    return best_p


def _voronoi_refine(params, deadline, bounds_list):
    """
    Refine using Voronoi-based radius estimation.
    For each circle, compute the maximum radius allowed by its Voronoi cell.
    """
    p = params.reshape(N, 3).copy()
    centers = p[:, :2]
    
    try:
        vor = Voronoi(centers)
    except Exception:
        return params
    
    # Compute Voronoi vertices
    for i in range(N):
        # Get vertices of the Voronoi cell for point i
        if i in vor.point_region:
            region_idx = vor.point_region[i]
            vertices = vor.regions[region_idx]
            if len(vertices) == 0 or -1 in vertices:
                continue
            
            vpts = vor.vertices[vertices]
            
            # Distance from center to each Voronoi vertex
            dists = np.linalg.norm(vpts - centers[i], axis=1)
            min_dist = np.min(dists)
            
            # Also check distance to walls
            wall_dist = np.min([centers[i][0], 1-centers[i][0], centers[i][1], 1-centers[i][1]])
            
            # Maximum radius is half the minimum distance to Voronoi vertex or wall
            max_r = 0.5 * np.min([min_dist, wall_dist])
            
            # Update radius if it's smaller than current
            if p[i, 2] < max_r:
                p[i, 2] = max_r
    
    # Enforce validity after radius update
    p = _enforce_validity(p)
    
    return p.reshape(-1)


def _adaptive_penalty_refine(params, deadline, bounds_list):
    """
    Adaptive penalty refinement: start with low penalty, increase gradually.
    This helps find better solutions by not being too constrained initially.
    """
    lo, hi = bounds_list
    best_params = params.copy()
    best_score = _score_valid(_enforce_validity(params))
    
    penalties = [100.0, 500.0, 2000.0, 5000.0, 10000.0, 50000.0]
    
    for pen in penalties:
        if time.time() > deadline - 3:
            break
        
        def obj(p, pen=pen):
            return _penalty_objective(p, penalty=pen)
        
        def grad(p, pen=pen):
            return _penalty_grad(p, penalty=pen)
        
        try:
            res = minimize(obj, best_params, jac=grad, method='L-BFGS-B',
                           bounds=bounds_list,
                           options={'maxiter': 100, 'maxfun': 200,
                                    'ftol': 1e-12, 'gtol': 1e-8})
            new_params = res.x
            new_score = _score_valid(_enforce_validity(new_params))
            
            if new_score > best_score:
                best_score = new_score
                best_params = new_params.copy()
        except Exception:
            pass
    
    return best_params


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE

    lo, hi = _bounds()
    bounds_list = list(zip(lo, hi))

    best_params = None
    best_score = -np.inf

    initializers = [
        lambda s: _voronoi_initial(s),
        lambda s: _hex_initial(s),
        lambda s: _grid_initial(s),
    ]

    seeds = [42, 123, 7, 999, 555, 2024, 31415, 271828, 1234, 5678,
             11, 22, 33, 44, 55, 66, 77, 88, 99, 100]
    
    for seed in seeds:
        if time.time() > deadline - 5:
            break
        for init_fn in initializers:
            if time.time() > deadline - 5:
                break
            params = init_fn(seed)
            rng = np.random.RandomState(seed + 1000)

            # Phase 1: Adaptive penalty L-BFGS-B (coarse to fine)
            if time.time() < deadline - 10:
                params = _adaptive_penalty_refine(params, deadline, bounds_list)
            
            # Phase 2: Delaunay-based SLSQP refinement (structural, precise)
            if time.time() < deadline - 5:
                params = _delaunay_refine(params, deadline, bounds_list)
            
            # Phase 3: Voronoi-based radius refinement
            if time.time() < deadline - 3:
                params = _voronoi_refine(params, deadline, bounds_list)
            
            # Phase 4: Shake and refine to escape local minima
            if time.time() < deadline - 2:
                time_left = max(5.0, deadline - time.time() - 2)
                params = _shake_and_refine(params.reshape(N, 3), 
                                          steps=3000, 
                                          time_limit=time_left,
                                          seed=seed + 2000)
                params = params.reshape(-1)
            
            # Phase 5: Final high-precision polish
            if time.time() < deadline - 1:
                try:
                    res = minimize(lambda p: _penalty_objective(p, penalty=50000.0),
                                   params,
                                   jac=lambda p: _penalty_grad(p, penalty=50000.0),
                                   method='L-BFGS-B', bounds=bounds_list,
                                   options={'maxiter': 50, 'maxfun': 100,
                                            'ftol': 1e-14, 'gtol': 1e-10})
                    params = res.x
                except Exception:
                    pass

            # Enforce validity and score
            valid_p = _enforce_validity(params)
            score = _score_valid(valid_p)

            if score > best_score:
                best_score = score
                best_params = valid_p.copy()

    if best_params is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-4)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    return best_params
