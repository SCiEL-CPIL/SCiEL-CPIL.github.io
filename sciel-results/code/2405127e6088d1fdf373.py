import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Voronoi
from scipy.spatial import Delaunay

N = 26
EPS = 1e-8
DEADLINE = 145.0  # safety margin before 150s kill


def _penalty_objective(params, penalty=1000.0):
    """Penalty objective: minimize -sum(r) + penalty*(overlaps + wall violations)."""
    p = params.reshape(N, 3)
    x = p[:, 0]
    y = p[:, 1]
    r = p[:, 2]

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
    r = p[:, 2]
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


def _voronoi_initial(seed):
    """Initialize using Voronoi cells to estimate max possible radii."""
    rng = np.random.RandomState(seed)
    points = rng.uniform(0.05, 0.95, size=(N, 2))
    
    try:
        vor = Voronoi(points)
    except Exception:
        # Fallback to grid if Voronoi fails
        return _grid_initial(seed)
    
    radii = np.zeros(N)
    for i, p in enumerate(points):
        # Distance to nearest neighbor
        dists = np.linalg.norm(points - p, axis=1)
        dists[i] = np.inf
        min_dist_neighbor = np.min(dists)
        
        # Distance to walls
        wall_dist = np.min([p[0], 1-p[0], p[1], 1-p[1]])
        
        # Initial radius is half the min distance, with a small safety factor
        radii[i] = 0.5 * np.min([min_dist_neighbor, wall_dist]) * 0.95
        
    params = np.empty(3 * N)
    params[0::3] = points[:, 0]
    params[1::3] = points[:, 1]
    params[2::3] = radii
    
    # Small jitter to break symmetry
    params += rng.uniform(-1e-5, 1e-5, size=params.size)
    return params

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

    # Clamp to walls
    wall = np.minimum(np.minimum(p[:, 0], 1.0 - p[:, 0]),
                      np.minimum(p[:, 1], 1.0 - p[:, 1]))
    p[:, 2] = np.minimum(p[:, 2], np.maximum(wall, 1e-6))

    # Iteratively resolve overlaps by shrinking the smaller circle
    for _ in range(100):
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
            # Shrink the smaller circle by ov (or split if equal)
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


def _score_valid(p):
    """Compute sum of radii for a valid configuration."""
    return np.sum(p[:, 2])


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
        
        # Score
        s_new = _score_valid(p_new)
        
        if s_new > best_score:
            best_score = s_new
            best_p = p_new
            
    return best_p

def _slsqp_refine(p, time_limit=10.0, seed=42):
    """
    Iterative SLSQP refinement based on contact graph.
    """
    best_p = p.copy()
    best_score = _score_valid(best_p)
    rng = np.random.RandomState(seed)
    start = time.time()
    
    # We will iterate a few times, each time identifying active constraints
    # and solving for max sum of radii subject to those constraints.
    for iteration in range(20):
        if time.time() > start + time_limit:
            break
        
        x = best_p[:, 0]
        y = best_p[:, 1]
        r = best_p[:, 2]
        
        # Identify active constraints (contacts)
        # 1. Wall contacts
        # 2. Pairwise contacts
        
        # We define "active" as being within a small tolerance of touching
        tol = 1e-5
        
        # Constraints list for SLSQP
        constraints = []
        
        # Wall constraints
        # x_i - r_i >= 0  => x_i - r_i - 0 >= 0
        # 1 - x_i - r_i >= 0
        # y_i - r_i >= 0
        # 1 - y_i - r_i >= 0
        
        # Pairwise constraints
        # dist(i,j) - r_i - r_j >= 0
        
        # Instead of adding ALL constraints, we add only active ones to speed up SLSQP
        # and help it focus on the "rigid" parts of the packing.
        
        # However, SLSQP needs a consistent set. If we drop a constraint that becomes active later,
        # we might violate it. A safer approach for "contact graph" is to include all 
        # constraints that are "close" to active, or just all of them if N is small enough.
        # N=26 => 325 pairs + 104 walls = 429 constraints. This is manageable for SLSQP.
        
        # Let's try adding all constraints but using a tighter tolerance for "activity" 
        # in the initial guess? No, SLSQP handles inequality constraints.
        
        # Objective: Maximize sum(r) => Minimize -sum(r)
        def obj(params):
            return -np.sum(params[2::3])
            
        def obj_grad(params):
            g = np.zeros_like(params)
            g[2::3] = -1.0
            return g
            
        # Constraints
        # Format: {'type': 'ineq', 'fun': lambda x: ..., 'jac': lambda x: ...}
        # fun(x) >= 0
        
        cons = []
        
        # Walls
        for i in range(N):
            # x_i - r_i >= 0
            def wall_x1(params, i=i):
                return params[3*i] - params[3*i+2]
            def wall_x1_jac(params, i=i):
                g = np.zeros_like(params)
                g[3*i] = 1.0
                g[3*i+2] = -1.0
                return g
            cons.append({'type': 'ineq', 'fun': wall_x1, 'jac': wall_x1_jac})
            
            # 1 - x_i - r_i >= 0
            def wall_x2(params, i=i):
                return 1.0 - params[3*i] - params[3*i+2]
            def wall_x2_jac(params, i=i):
                g = np.zeros_like(params)
                g[3*i] = -1.0
                g[3*i+2] = -1.0
                return g
            cons.append({'type': 'ineq', 'fun': wall_x2, 'jac': wall_x2_jac})
            
            # y_i - r_i >= 0
            def wall_y1(params, i=i):
                return params[3*i+1] - params[3*i+2]
            def wall_y1_jac(params, i=i):
                g = np.zeros_like(params)
                g[3*i+1] = 1.0
                g[3*i+2] = -1.0
                return g
            cons.append({'type': 'ineq', 'fun': wall_y1, 'jac': wall_y1_jac})
            
            # 1 - y_i - r_i >= 0
            def wall_y2(params, i=i):
                return 1.0 - params[3*i+1] - params[3*i+2]
            def wall_y2_jac(params, i=i):
                g = np.zeros_like(params)
                g[3*i+1] = -1.0
                g[3*i+2] = -1.0
                return g
            cons.append({'type': 'ineq', 'fun': wall_y2, 'jac': wall_y2_jac})
            
        # Pairs
        iu = np.triu_indices(N, k=1)
        for k in range(len(iu[0])):
            i = iu[0][k]
            j = iu[1][k]
            
            def pair_dist(params, i=i, j=j):
                dx = params[3*i] - params[3*j]
                dy = params[3*i+1] - params[3*j+1]
                dist = np.sqrt(dx*dx + dy*dy)
                return dist - params[3*i+2] - params[3*j+2]
                
            def pair_dist_jac(params, i=i, j=j):
                g = np.zeros_like(params)
                dx = params[3*i] - params[3*j]
                dy = params[3*i+1] - params[3*j+1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < 1e-10:
                    dist = 1e-10
                g[3*i] = dx / dist
                g[3*j] = -dx / dist
                g[3*i+1] = dy / dist
                g[3*j+1] = -dy / dist
                g[3*i+2] = -1.0
                g[3*j+2] = -1.0
                return g
                
            cons.append({'type': 'ineq', 'fun': pair_dist, 'jac': pair_dist_jac})

        # Bounds
        lo, hi = _bounds()
        bounds = list(zip(lo, hi))
        
        # Solve
        try:
            res = minimize(
                obj,
                best_p.flatten(),
                method='SLSQP',
                jac=obj_grad,
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 100, 'ftol': 1e-12}
            )
            
            if res.success:
                p_new = res.x.reshape(N, 3)
                # Enforce validity just in case SLSQP drifted slightly
                p_new = _enforce_validity(p_new)
                s_new = _score_valid(p_new)
                
                if s_new > best_score:
                    best_score = s_new
                    best_p = p_new
                else:
                    # If no improvement, maybe shake a bit?
                    # Or break if we are stagnating?
                    pass
                    
        except Exception as e:
            # If SLSQP fails, try a small shake and retry? Or just continue.
            pass
            
    return best_p


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE

    lo, hi = _bounds()
    bounds_list = list(zip(lo, hi))

    best_params = None
    best_score = -np.inf

    # Multi-start with different initializations
    initializers = [
        lambda s: _voronoi_initial(s),
        lambda s: _hex_initial(s),
        lambda s: _grid_initial(s),
    ]

    # Use more seeds to explore the landscape better
    seeds = [42, 123, 7, 999, 555, 2024, 31415, 271828, 1234, 5678, 
             11, 22, 33, 44, 55, 66, 77, 88, 99, 100]
    
    for seed in seeds:
        if time.time() > deadline - 5:
            break
        for init_fn in initializers:
            if time.time() > deadline - 5:
                break
            
            x0 = init_fn(seed)
            
            # First, enforce validity to get a safe starting point
            x0 = _enforce_validity(x0).flatten()
            
            # Run SLSQP refinement (Contact Graph Aware)
            # Allocate time per refinement
            time_per_refine = 5.0 # 5 seconds per SLSQP run
            if time.time() + time_per_refine > deadline:
                 continue
                
            try:
                p_refined = _slsqp_refine(x0.reshape(N, 3), time_limit=time_per_refine, seed=seed)
                s = _score_valid(p_refined)
                
                if s > best_score:
                    best_score = s
                    best_params = p_refined.copy()
                    
            except Exception:
                continue

    # If we didn't find any valid solution, use a fallback
    if best_params is None:
        best_params = _enforce_validity(_grid_initial(42)).reshape(N, 3)

    # Final refinement: Use stochastic shake and solve to escape local minima
    # Allocate remaining time to this step
    remaining_time = deadline - time.time() - 2.0 # 2s safety margin
    if remaining_time > 5.0:
        # Run shake and refine
        refine_seeds = [1, 2, 3, 4, 5]
        for r_seed in refine_seeds:
            if time.time() > deadline - 2.0:
                break
            t_limit = (deadline - time.time() - 2.0) / len(refine_seeds)
            if t_limit < 1.0:
                break
            p_refined = _shake_and_refine(best_params, steps=10000, time_limit=t_limit, seed=r_seed)
            s_refined = _score_valid(p_refined)
            if s_refined > best_score:
                best_score = s_refined
                best_params = p_refined
            # After shaking, try SLSQP again on the new start?
            # This might be too slow. Let's just stick to shake.
            # Or maybe a quick SLSQP if time permits?
            if time.time() < deadline - 10:
                 p_slsqp = _slsqp_refine(best_params, time_limit=5.0, seed=r_seed+100)
                 s_slsqp = _score_valid(p_slsqp)
                 if s_slsqp > best_score:
                     best_score = s_slsqp
                     best_params = p_slsqp

    # Return in the required format: (x, y, radius)
    result = best_params.copy()
    return result
