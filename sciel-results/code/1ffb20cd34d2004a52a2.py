"""Circle packing in unit square: 26 disjoint circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay
import scipy.spatial as sp

N = 26
EPS = 1e-8
DEADLINE = 145.0  # Safety margin before 150s kill

def _elapsed(start):
    return time.time() - start

def _remaining(start, deadline):
    return deadline - time.time()

def _softplus(x, k=100.0):
    """Smooth approximation to max(0, x)."""
    # Avoid overflow
    x = np.clip(x, -50, 50)
    return np.log1p(np.exp(k * x)) / k

def _softplus_grad(x, k=100.0):
    """Gradient of softplus."""
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-k * x))

def _validate(p):
    """Check if configuration is valid. Returns (is_valid, score)."""
    x = p[:, 0]
    y = p[:, 1]
    r = p[:, 2]
    
    # Wall checks
    if np.any(x < -1e-9) or np.any(x > 1 + 1e-9) or np.any(y < -1e-9) or np.any(y > 1 + 1e-9):
        return False, 0.0
    if np.any(r < 1e-9):
        return False, 0.0
    if np.any(x < r - 1e-9) or np.any(x > 1 - r + 1e-9) or np.any(y < r - 1e-9) or np.any(y > 1 - r + 1e-9):
        return False, 0.0
        
    # Pairwise checks
    dists = sp.distance_matrix(x[:, None], x[:, None]) + sp.distance_matrix(y[:, None], y[:, None])
    dists = np.sqrt(dists)
    min_dists = np.min(dists, axis=1) # Not quite right, need pairwise
    # Correct pairwise distance check
    iu = np.triu_indices(N, k=1)
    dx = x[iu[0]] - x[iu[1]]
    dy = y[iu[0]] - y[iu[1]]
    dist = np.sqrt(dx*dx + dy*dy)
    if np.any(dist < r[iu[0]] + r[iu[1]] - 1e-9):
        return False, 0.0
        
    return True, np.sum(r)

def _fix_boundaries(p):
    """Clip positions and radii to be within bounds, shrinking radii if necessary."""
    p = p.copy()
    x = p[:, 0]
    y = p[:, 1]
    r = p[:, 2]
    
    # Clip positions to [0, 1]
    x = np.clip(x, 0, 1)
    y = np.clip(y, 0, 1)
    
    # Max radius based on walls
    r_max = np.minimum(np.minimum(x, 1-x), np.minimum(y, 1-y))
    r = np.minimum(r, r_max)
    r = np.maximum(r, 1e-6)
    
    p[:, 0] = x
    p[:, 1] = y
    p[:, 2] = r
    
    return p

def _get_contact_edges(p, threshold=1.1):
    """Get indices of pairs that are close to touching (Delaunay neighbors within threshold)."""
    x = p[:, 0]
    y = p[:, 1]
    r = p[:, 2]
    
    points = np.column_stack((x, y))
    try:
        tri = Delaunay(points)
    except Exception:
        return np.array([], dtype=int), np.array([], dtype=int)
    
    # Get unique edges from triangles
    edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i+1, 3):
                u, v = simplex[i], simplex[j]
                if u > v: u, v = v, u
                edges.add((u, v))
                
    if not edges:
        return np.array([], dtype=int), np.array([], dtype=int)
        
    edge_list = list(edges)
    u = np.array([e[0] for e in edge_list])
    v = np.array([e[1] for e in edge_list])
    
    # Filter by distance
    dx = x[u] - x[v]
    dy = y[u] - y[v]
    dist = np.sqrt(dx*dx + dy*dy)
    sum_r = r[u] + r[v]
    
    # Keep edges where dist <= sum_r * threshold
    mask = dist <= sum_r * threshold
    u = u[mask]
    v = v[mask]
    
    return u, v

def _objective_penalty(p, contact_u, contact_v, penalty=1000.0, k=50.0):
    """
    Objective: -sum(r) + penalty * (overlaps + wall_violations)
    Using softplus for smoothness.
    """
    x = p[:, 0]
    y = p[:, 1]
    r = p[:, 2]
    
    score = -np.sum(r)
    
    # Wall violations
    # x < r, x > 1-r, y < r, y > 1-r
    wx_left = r - x
    wx_right = r - (1 - x)
    wy_bottom = r - y
    wy_top = r - (1 - y)
    
    wall_pen = np.sum(_softplus(wx_left, k)) + np.sum(_softplus(wx_right, k)) + \
               np.sum(_softplus(wy_bottom, k)) + np.sum(_softplus(wy_top, k))
    
    # Pairwise overlaps (only for contact edges)
    if len(contact_u) > 0:
        dx = x[contact_u] - x[contact_v]
        dy = y[contact_u] - y[contact_v]
        dist = np.sqrt(dx*dx + dy*dy)
        overlap = r[contact_u] + r[contact_v] - dist
        pair_pen = np.sum(_softplus(overlap, k))
    else:
        pair_pen = 0.0
        
    return score + penalty * (wall_pen + pair_pen)

def _objective_grad(p, contact_u, contact_v, penalty=1000.0, k=50.0):
    """Analytic gradient of the penalty objective."""
    x = p[:, 0]
    y = p[:, 1]
    r = p[:, 2]
    
    g = np.zeros_like(p)
    
    # d(-sum r)/dr = -1
    g[:, 2] = -1.0
    
    # Wall gradients
    # d(_softplus(r-x))/dr = _softplus_grad(r-x)
    # d(_softplus(r-x))/dx = -_softplus_grad(r-x)
    
    wx_left = r - x
    g_left = _softplus_grad(wx_left, k)
    g[:, 2] += penalty * g_left
    g[:, 0] -= penalty * g_left
    
    wx_right = r - (1 - x)
    g_right = _softplus_grad(wx_right, k)
    g[:, 2] += penalty * g_right
    g[:, 0] += penalty * g_right
    
    wy_bottom = r - y
    g_bottom = _softplus_grad(wy_bottom, k)
    g[:, 2] += penalty * g_bottom
    g[:, 1] -= penalty * g_bottom
    
    wy_top = r - (1 - y)
    g_top = _softplus_grad(wy_top, k)
    g[:, 2] += penalty * g_top
    g[:, 1] += penalty * g_top
    
    # Pairwise gradients
    if len(contact_u) > 0:
        dx = x[contact_u] - x[contact_v]
        dy = y[contact_u] - y[contact_v]
        dist = np.sqrt(dx*dx + dy*dy)
        dist_safe = np.maximum(dist, 1e-10)
        
        overlap = r[contact_u] + r[contact_v] - dist
        grad_sp = _softplus_grad(overlap, k)
        
        # d(overlap)/dx_u = -dx/dist
        # d(overlap)/dx_v = +dx/dist
        # d(overlap)/dr_u = 1
        # d(overlap)/dr_v = 1
        
        grad_x_u = -dx / dist_safe
        grad_x_v = dx / dist_safe
        grad_y_u = -dy / dist_safe
        grad_y_v = dy / dist_safe
        
        # Accumulate gradients
        np.add.at(g[:, 2], contact_u, penalty * grad_sp)
        np.add.at(g[:, 2], contact_v, penalty * grad_sp)
        
        np.add.at(g[:, 0], contact_u, penalty * grad_sp * grad_x_u)
        np.add.at(g[:, 0], contact_v, penalty * grad_sp * grad_x_v)
        
        np.add.at(g[:, 1], contact_u, penalty * grad_sp * grad_y_u)
        np.add.at(g[:, 1], contact_v, penalty * grad_sp * grad_y_v)
        
    return g

def _initial_hex(seed):
    """Hexagonal grid initialization."""
    rng = np.random.RandomState(seed)
    points = []
    # Generate a dense hex grid
    for i in range(-2, 8):
        for j in range(-2, 8):
            x = i + 0.5 * (j % 2)
            y = j * np.sqrt(3) / 2
            points.append((x, y))
            
    points = np.array(points)
    # Center and scale to fit in [0,1]^2 with margin
    points -= np.mean(points, axis=0)
    max_coord = np.max(np.abs(points))
    scale = 0.45 / max_coord
    points *= scale
    points += 0.5
    
    # Select N points closest to center
    dists = np.sum((points - 0.5)**2, axis=1)
    indices = np.argsort(dists)[:N]
    
    p = points[indices]
    # Estimate radii
    radii = np.full(N, 0.05)
    
    params = np.empty(3 * N)
    params[0::3] = p[:, 0]
    params[1::3] = p[:, 1]
    params[2::3] = radii
    
    # Jitter
    params += rng.uniform(-1e-4, 1e-4, size=params.size)
    return params

def _initial_grid(seed):
    """Grid initialization."""
    rng = np.random.RandomState(seed)
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side - 1e-4
            cells.append((x, y, r))
            
    cells = cells[:N]
    arr = np.array(cells, dtype=float).reshape(-1)
    arr += rng.uniform(-1e-5, 1e-5, size=arr.size)
    return arr

def _initial_random(seed):
    """Random initialization."""
    rng = np.random.RandomState(seed)
    x = rng.uniform(0.1, 0.9, N)
    y = rng.uniform(0.1, 0.9, N)
    r = np.full(N, 0.05)
    
    params = np.empty(3 * N)
    params[0::3] = x
    params[1::3] = y
    params[2::3] = r
    
    return params

def _delaunay_polish(p, max_iter=50, time_limit=10.0):
    """
    Refine the configuration by enforcing contact constraints on Delaunay edges.
    This helps to tighten the packing.
    """
    start = time.time()
    p = p.copy()
    
    for _ in range(max_iter):
        if time.time() - start > time_limit:
            break
            
        u, v = _get_contact_edges(p, threshold=1.05)
        
        if len(u) == 0:
            break
            
        # Define a local optimization problem:
        # Maximize sum(r) subject to d_ij >= r_i + r_j for all i,j
        # and wall constraints.
        # We use L-BFGS-B with the softplus penalty again, but with a higher penalty
        # and focused on the contact graph.
        
        def obj(p):
            return _objective_penalty(p, u, v, penalty=5000.0, k=100.0)
            
        def grad(p):
            return _objective_grad(p, u, v, penalty=5000.0, k=100.0)
            
        bounds = [(0, 1), (0, 1), (1e-6, 0.5)] * N
        
        try:
            res = minimize(obj, p, jac=grad, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 50, 'maxfun': 100, 'ftol': 1e-12, 'gtol': 1e-8})
            p = res.x
        except Exception:
            break
            
        # Check for improvement
        # If no improvement, break
        # (Simple check: if objective didn't decrease significantly)
        
    return p

def _enforce_validity(params):
    """Enforce all constraints by shrinking radii and adjusting positions."""
    p = params.reshape(N, 3).copy()
    p[:, 0] = np.clip(p[:, 0], 0.0, 1.0)
    p[:, 1] = np.clip(p[:, 1], 0.0, 1.0)
    p[:, 2] = np.clip(p[:, 2], 1e-6, 0.5)

    # Clamp to walls
    wall = np.minimum(np.minimum(p[:, 0], 1.0 - p[:, 0]),
                      np.minimum(p[:, 1], 1.0 - p[:, 1]))
    p[:, 2] = np.minimum(p[:, 2], np.maximum(wall, 1e-6))

    # Iteratively resolve overlaps by shrinking the smaller circle
    for _ in range(200):
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
        # Re-clamp to walls
        wall = np.minimum(np.minimum(p[:, 0], 1.0 - p[:, 0]),
                          np.minimum(p[:, 1], 1.0 - p[:, 1]))
        p[:, 2] = np.minimum(p[:, 2], np.maximum(wall, 1e-6))

    return p

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE

    lo = np.zeros(3 * N)
    hi = np.zeros(3 * N)
    for i in range(N):
        lo[3*i] = 0.0
        hi[3*i] = 1.0
        lo[3*i+1] = 0.0
        hi[3*i+1] = 1.0
        lo[3*i+2] = 1e-6
        hi[3*i+2] = 0.5
    bounds_list = list(zip(lo, hi))

    best_params = None
    best_score = -np.inf

    initializers = [
        _initial_hex,
        _initial_grid,
        _initial_random
    ]

    seeds = [42, 123, 7, 999, 555, 2024, 31415, 271828, 1234, 5678]
    
    for seed in seeds:
        if time.time() > deadline - 10:
            break
            
        for init_fn in initializers:
            if time.time() > deadline - 10:
                break
                
            params = init_fn(seed)
            
            # Multi-stage optimization
            # Stage 1: Low penalty, find a feasible region
            # Stage 2: Medium penalty, refine
            # Stage 3: High penalty, tighten
            
            for pen, k in [(100.0, 20.0), (1000.0, 50.0), (5000.0, 100.0)]:
                if time.time() > deadline - 5:
                    break
                    
                # Update contact edges
                u, v = _get_contact_edges(params, threshold=1.5)
                
                def obj(p):
                    return _objective_penalty(p, u, v, penalty=pen, k=k)
                    
                def grad(p):
                    return _objective_grad(p, u, v, penalty=pen, k=k)
                    
                try:
                    res = minimize(obj, params, jac=grad, method='L-BFGS-B', bounds=bounds_list,
                                   options={'maxiter': 100, 'maxfun': 200, 'ftol': 1e-12, 'gtol': 1e-8})
                    params = res.x
                except Exception:
                    pass
                
            # Delaunay Polish
            if time.time() < deadline - 5:
                params = _delaunay_polish(params, max_iter=20, time_limit=5.0)
                
            # Enforce validity
            valid_p = _enforce_validity(params)
            score = _score_valid(valid_p)
            
            if score > best_score:
                best_score = score
                best_params = valid_p.copy()

    if best_params is None:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-4)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    return best_params

def _score_valid(p):
    return np.sum(p[:, 2])

