"""Circle packing in unit square: 26 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay
from cma import CMAES

N = 26

# EVOLVE-BLOCK-START

def _validate(xyz):
    """Check if configuration is valid: circles inside unit square and non-overlapping."""
    n = xyz.shape[0]
    x = xyz[:, 0]
    y = xyz[:, 1]
    r = xyz[:, 2]
    if np.any(r < -1e-9):
        return False
    if np.any(x - r < -1e-7) or np.any(x + r > 1 + 1e-7):
        return False
    if np.any(y - r < -1e-7) or np.any(y + r > 1 + 1e-7):
        return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - 1e-7:
                return False
    return True

def _get_delaunay_edges(centers):
    """Get Delaunay triangulation edges for the given centers."""
    try:
        tri = Delaunay(centers)
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                a, b = simplex[i], simplex[(i + 1) % 3]
                if a > b:
                    a, b = b, a
                edges.add((a, b))
        return list(edges)
    except Exception:
        return [(i, j) for i in range(N) for j in range(i + 1, N)]

def _enforce_validity(params):
    """
    Enforce all constraints by shrinking radii and clipping positions.
    Returns a valid configuration.
    """
    p = params.reshape(N, 3).copy()
    p[:, 0] = np.clip(p[:, 0], 0.0, 1.0)
    p[:, 1] = np.clip(p[:, 1], 0.0, 1.0)
    p[:, 2] = np.clip(p[:, 2], 1e-6, 0.5)
    
    # Clip radii to fit within walls
    wall = np.minimum(np.minimum(p[:, 0], 1.0 - p[:, 0]),
                      np.minimum(p[:, 1], 1.0 - p[:, 1]))
    p[:, 2] = np.minimum(p[:, 2], np.maximum(wall, 1e-6))
    
    # Iteratively resolve overlaps
    for _ in range(300):
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
            # Shrink the smaller circle preferentially
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
        # Re-clip to walls
        wall = np.minimum(np.minimum(p[:, 0], 1.0 - p[:, 0]),
                          np.minimum(p[:, 1], 1.0 - p[:, 1]))
        p[:, 2] = np.minimum(p[:, 2], np.maximum(wall, 1e-6))
    
    return p

def _objective_cma(params, penalty_weight=1000.0):
    """
    Objective function for CMA-ES.
    Minimizes -sum(r) + penalty.
    """
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    penalty = 0.0
    
    # Boundary penalties
    pen_x_lo = np.maximum(0, -(x - r))
    pen_x_hi = np.maximum(0, x + r - 1)
    pen_y_lo = np.maximum(0, -(y - r))
    pen_y_hi = np.maximum(0, y + r - 1)
    pen_r = np.maximum(0, -r)
    penalty += np.sum(pen_x_lo ** 2) + np.sum(pen_x_hi ** 2)
    penalty += np.sum(pen_y_lo ** 2) + np.sum(pen_y_hi ** 2)
    penalty += np.sum(pen_r ** 2)
    
    # Full pairwise overlap penalties
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            overlap = r[i] + r[j] - dist
            if overlap > 0:
                penalty += overlap ** 2
    
    return -np.sum(r) + penalty_weight * penalty

def _cma_es_optimize(xyz0, time_limit=5.0, penalty_weight=1000.0):
    """
    Optimize using CMA-ES.
    """
    start_time = time.time()
    x0 = xyz0.flatten()
    
    # Bounds for CMA-ES
    # x, y in [0, 1], r in [0, 0.5]
    # CMA-ES in cma library handles bounds via transformation or clipping
    # We'll use a transformation to ensure bounds are respected or just clip and rely on penalty
    
    options = {
        'maxiter': 500,
        'tolopt': 1e-6,
        'tolfun': 1e-6,
        'verbose': -9,
        'seed': 42
    }
    
    # We need to define the bounds. CMA-ES supports bounds.
    # Shape (N, 3) -> flattened to 3N
    # x: [0, 1], y: [0, 1], r: [0, 0.5]
    bounds = np.zeros((N, 3))
    bounds[:, 0] = [0, 1]
    bounds[:, 1] = [0, 1]
    bounds[:, 2] = [0, 0.5]
    
    # Flatten bounds for cma library
    # cma library expects bounds as (N, 2) or similar? 
    # Actually, cma.CMAES takes 'bounds' as a list of tuples or array.
    # Let's check cma documentation. 
    # cma.CMAES(x0, sigma0, bounds=(lower, upper))
    # Lower: [0, 0, 0, ...]
    # Upper: [1, 1, 0.5, ...]
    
    lower = np.zeros(3 * N)
    upper = np.zeros(3 * N)
    for i in range(N):
        lower[3*i] = 0.0
        upper[3*i] = 1.0
        lower[3*i+1] = 0.0
        upper[3*i+1] = 1.0
        lower[3*i+2] = 0.0
        upper[3*i+2] = 0.5
        
    es = CMAES(x0, 0.05, bounds=(lower, upper), options=options)
    
    best_x = x0.copy()
    best_f = _objective_cma(best_x, penalty_weight)
    
    while not es.stop():
        if time.time() - start_time > time_limit:
            break
        sol = es.ask()
        f = _objective_cma(sol, penalty_weight)
        es.tell(sol, f)
        if f < best_f:
            best_f = f
            best_x = sol.copy()
            
    return best_x.reshape(N, 3)

def _polish_full(xyz, time_budget=5.0):
    """
    Final high-precision polish using full pairwise constraints.
    Uses very tight tolerances and high penalty weights for precision.
    """
    start_time = time.time()
    params = xyz.flatten()
    best_params = params.copy()
    best_obj = _objective_cma(best_params, 1000.0)
    
    bounds = []
    for i in range(N):
        bounds.append((0, 1))
        bounds.append((0, 1))
        bounds.append((1e-6, 0.5))
    
    for penalty_weight in [1000.0, 5000.0, 20000.0]:
        if time.time() - start_time > time_budget:
            break
        
        def obj(p):
            return _objective_cma(p, penalty_weight)
        
        try:
            result = minimize(obj, best_params, method='L-BFGS-B', bounds=bounds,
                             options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12})
            if result.fun < best_obj:
                best_obj = result.fun
                best_params = result.x
        except Exception:
            pass
    
    return best_params.reshape(N, 3)

def _initial_config_greedy():
    """
    Greedy initialization: place circles one by one at the position
    that allows the largest radius without overlapping existing circles
    or going out of bounds.
    """
    centers = []
    radii = []
    x = np.zeros(N)
    y = np.zeros(N)
    r = np.zeros(N)
    
    # Grid of potential centers
    grid_res = 50
    xs = np.linspace(0.05, 0.95, grid_res)
    ys = np.linspace(0.05, 0.95, grid_res)
    
    for i in range(N):
        best_r = 0
        best_x, best_y = 0.5, 0.5
        
        for gx in xs:
            for gy in ys:
                # Max radius allowed by walls
                r_wall = min(gx, 1 - gx, gy, 1 - gy)
                if r_wall <= 0:
                    continue
                
                # Max radius allowed by existing circles
                r_max = r_wall
                for j in range(len(centers)):
                    cx, cy, cr = centers[j]
                    dist = np.sqrt((gx - cx)**2 + (gy - cy)**2)
                    r_allowed = dist - cr
                    if r_allowed < r_max:
                        r_max = r_allowed
                
                if r_max > best_r:
                    best_r = r_max
                    best_x, best_y = gx, gy
        
        if best_r < 0.01:
            # Fallback: random placement
            best_x = np.random.uniform(0.1, 0.9)
            best_y = np.random.uniform(0.1, 0.9)
            best_r = 0.05
            
        x[i] = best_x
        y[i] = best_y
        r[i] = best_r
        centers.append((best_x, best_y, best_r))
        
    return np.column_stack([x, y, r])

def _initial_config_random():
    """Random initial configuration."""
    x = np.random.uniform(0.1, 0.9, N)
    y = np.random.uniform(0.1, 0.9, N)
    r = np.random.uniform(0.05, 0.1, N)
    for i in range(N):
        x[i] = np.clip(x[i], r[i], 1 - r[i])
        y[i] = np.clip(y[i], r[i], 1 - r[i])
    return np.column_stack([x, y, r])

def construct():
    """
    Construct 26 circles with disjoint interiors inside [0,1]^2,
    maximizing the sum of radii.
    """
    global_start = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    best_xyz = _initial_config_greedy()
    best_score = np.sum(best_xyz[:, 2])
    
    # List of initial configurations to try
    initial_configs = []
    
    # 1. Greedy
    initial_configs.append(_initial_config_greedy())
    
    # 2. Random
    np.random.seed(42)
    for _ in range(3):
        initial_configs.append(_initial_config_random())
        
    # 3. Grid
    side = 6
    positions = []
    for i in range(side):
        for j in range(side):
            if len(positions) >= N:
                break
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side * 0.8
            positions.append([x, y, r])
    initial_configs.append(np.array(positions[:N]))
    
    # Optimize from each initial configuration
    for idx, xyz0 in enumerate(initial_configs):
        elapsed = time.time() - global_start
        remaining = deadline - elapsed
        if remaining < 10.0:
            break
        
        # Phase 1: CMA-ES
        cma_time = min(10.0, remaining * 0.6)
        if cma_time < 2.0:
            break
            
        try:
            xyz_cma = _cma_es_optimize(xyz0, time_limit=cma_time, penalty_weight=1000.0)
            
            # Phase 2: Polish
            elapsed = time.time() - global_start
            remaining = deadline - elapsed
            polish_time = min(5.0, remaining * 0.4)
            if polish_time < 1.0:
                xyz_opt = xyz_cma
            else:
                xyz_opt = _polish_full(xyz_cma, time_budget=polish_time)
            
            # Validate and fix
            if not _validate(xyz_opt):
                xyz_fixed = _enforce_validity(xyz_opt)
                xyz_opt = xyz_fixed
            
            score = np.sum(xyz_opt[:, 2])
            if score > best_score:
                best_score = score
                best_xyz = xyz_opt
        except Exception:
            pass
    
    # Final check
    if not _validate(best_xyz):
        best_xyz = _enforce_validity(best_xyz)
    
    if not _validate(best_xyz):
        best_xyz = _initial_config_greedy()
    
    return best_xyz

# EVOLVE-BLOCK-END
