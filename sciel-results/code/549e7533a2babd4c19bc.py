"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations

def _precompute_indices():
    """Precompute all (i,j,k) index triples for N points."""
    idxs = list(combinations(range(N), 3))
    return np.array(idxs)

_IDX = _precompute_indices()
_I = _IDX[:, 0]
_J = _IDX[:, 1]
_K = _IDX[:, 2]

def _min_triangle_area_vec(points):
    """
    Compute the minimum area of any triangle formed by three points.
    Vectorized for speed.
    points: (N, 2) array
    """
    pi = points[_I]
    pj = points[_J]
    pk = points[_K]
    
    v1x = pj[:, 0] - pi[:, 0]
    v1y = pj[:, 1] - pi[:, 1]
    v2x = pk[:, 0] - pi[:, 0]
    v2y = pk[:, 1] - pi[:, 1]
    
    cross = v1x * v2y - v1y * v2x
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _clamp_to_triangle(p):
    """Clamp a point to be inside the equilateral triangle (0,0), (1,0), (0.5, sqrt(3)/2)."""
    x, y = p
    h = np.sqrt(3) / 2
    eps = 1e-9
    
    y = np.clip(y, eps, h - eps)
    
    x_min_left = y / np.sqrt(3)
    x_max_right = 1 - y / np.sqrt(3)
    
    x = np.clip(x, x_min_left + eps, x_max_right - eps)
    
    return np.array([x, y])

def _clamp_all(points):
    """Clamp all points to the triangle."""
    return np.array([_clamp_to_triangle(p) for p in points])

def _nelder_mead_optimize(points, time_limit=5.0):
    """
    Perform Nelder-Mead optimization to maximize the minimum triangle area.
    """
    from scipy.optimize import minimize
    
    points = np.array(points, dtype=float)
    points = _clamp_all(points)
    
    best_points = points.copy()
    best_score = _min_triangle_area_vec(points)
    
    start_time = time.time()
    
    # Flatten points to 22-dimensional vector
    x0 = points.flatten()
    
    def objective(x):
        if time.time() - start_time > time_limit:
            return 0.0
        pts = x.reshape(N, 2)
        pts = _clamp_all(pts)
        return -_min_triangle_area_vec(pts)
    
    # Run Nelder-Mead
    try:
        result = minimize(
            objective,
            x0,
            method='Nelder-Mead',
            options={
                'maxiter': 500,
                'maxfev': 2000,
                'xatol': 1e-8,
                'fatol': 1e-10
            }
        )
        
        if result.success or result.fun < objective(x0):
            opt_pts = result.x.reshape(N, 2)
            opt_pts = _clamp_all(opt_pts)
            score = _min_triangle_area_vec(opt_pts)
            if score > best_score:
                best_score = score
                best_points = opt_pts
    except Exception:
        pass
    
    return best_points, best_score

def _simulated_annealing(start_points, time_limit=10.0):
    """
    Perform Simulated Annealing to maximize the minimum triangle area.
    """
    points = np.array(start_points, dtype=float)
    points = _clamp_all(points)
    current_score = _min_triangle_area_vec(points)
    
    best_points = points.copy()
    best_score = current_score
    
    # Parameters
    T = 0.01
    T_min = 1e-6
    alpha = 0.9995
    step_size = 0.05
    
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        if T < T_min:
            break
            
        # Pick a random point to perturb
        idx = np.random.randint(N)
        
        # Perturb with Gaussian noise
        noise = np.random.normal(0, step_size, 2)
        new_point = points[idx] + noise
        
        # Clamp the new point
        new_point = _clamp_to_triangle(new_point)
        
        # Update points
        old_point = points[idx].copy()
        points[idx] = new_point
        
        new_score = _min_triangle_area_vec(points)
        
        # Acceptance criterion
        delta = new_score - current_score
        if delta > 0 or np.random.random() < np.exp(delta / T):
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = points.copy()
        else:
            points[idx] = old_point
            
        # Cool down
        T *= alpha
        step_size *= 0.99995
        
    return best_points, best_score

def _generate_initial_configs():
    """Generate diverse initial configurations."""
    h = np.sqrt(3) / 2
    configs = []
    
    # 1. Symmetric configuration
    sym = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, h],
        [0.5, h/3],
        [0.25, h/6],
        [0.75, h/6],
        [0.5, h/2],
        [0.125, h/12],
        [0.875, h/12],
        [0.375, h/4],
        [0.625, h/4]
    ])
    configs.append(sym)
    
    # 2. Another symmetric configuration
    sym2 = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, h],
        [0.5, h/2],
        [0.25, h/4],
        [0.75, h/4],
        [0.5, h/6],
        [0.25, h/12],
        [0.75, h/12],
        [0.125, h/24],
        [0.875, h/24]
    ])
    configs.append(sym2)
    
    # 3. Grid-like configuration
    grid = np.array([
        [0.0, 0.0],
        [0.5, 0.0],
        [1.0, 0.0],
        [0.25, h/4],
        [0.75, h/4],
        [0.5, h/2],
        [0.125, h/8],
        [0.375, h/8],
        [0.625, h/8],
        [0.875, h/8],
        [0.5, h/6]
    ])
    configs.append(grid)
    
    # 4. Random configurations
    rng = np.random.default_rng(42)
    for i in range(15):
        pts = rng.random((N, 2))
        pts[:, 1] *= h
        pts = _clamp_all(pts)
        configs.append(pts)
        
    return configs

def _local_refine(points, time_limit=2.0):
    """
    Local refinement using coordinate descent + Nelder-Mead on pairs of points.
    """
    from scipy.optimize import minimize
    
    points = np.array(points, dtype=float)
    points = _clamp_all(points)
    best_points = points.copy()
    best_score = _min_triangle_area_vec(points)
    
    start_time = time.time()
    
    # Try optimizing each point individually
    for i in range(N):
        if time.time() - start_time > time_limit:
            break
            
        p = points[i].copy()
        
        def obj_single(x):
            pts = points.copy()
            pts[i] = _clamp_to_triangle(x)
            return -_min_triangle_area_vec(pts)
        
        try:
            res = minimize(obj_single, p, method='Nelder-Mead',
                          options={'maxiter': 50, 'maxfev': 100, 'xatol': 1e-7, 'fatol': 1e-9})
            if res.fun < obj_single(p):
                new_p = _clamp_to_triangle(res.x)
                points[i] = new_p
                score = _min_triangle_area_vec(points)
                if score > best_score:
                    best_score = score
                    best_points = points.copy()
        except Exception:
            pass
    
    return best_points, best_score

def construct():
    """
    Place 11 points on or inside a unit-area equilateral triangle to maximize
    the smallest area of a triangle formed by three points.
    """
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin before 150s limit
    
    best_points = None
    best_score = -1.0
    
    configs = _generate_initial_configs()
    
    # Phase 1: Quick Nelder-Mead on each config
    n_configs = len(configs)
    time_per_config = 4.0
    
    for i, config in enumerate(configs):
        if time.time() > deadline:
            break
            
        time_left = deadline - time.time()
        current_time_limit = min(time_per_config, time_left / (n_configs - i))
        
        if current_time_limit < 1.0:
            break
            
        try:
            opt_pts, score = _nelder_mead_optimize(config, time_limit=current_time_limit)
            if score > best_score:
                best_score = score
                best_points = opt_pts
        except Exception:
            continue
            
    # Phase 2: SA on the best found so far
    if best_points is not None and time.time() < deadline - 10:
        remaining_time = deadline - time.time() - 10
        if remaining_time > 2.0:
            try:
                opt_pts, score = _simulated_annealing(best_points, time_limit=remaining_time)
                if score > best_score:
                    best_score = score
                    best_points = opt_pts
            except Exception:
                pass
                
    # Phase 3: Final local refinement
    if best_points is not None and time.time() < deadline - 2:
        remaining_time = deadline - time.time() - 2
        if remaining_time > 1.0:
            try:
                opt_pts, score = _local_refine(best_points, time_limit=remaining_time)
                if score > best_score:
                    best_score = score
                    best_points = opt_pts
            except Exception:
                pass

    if best_points is None:
        best_points = configs[0]
        
    return best_points

# EVOLVE-BLOCK-END
