# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull, Delaunay
from scipy.optimize import minimize
import time
import itertools

# Precompute all triangle index combinations (286 triangles for 13 points)
N_POINTS = 13
N_TRI = 286
_TRI_IDX = np.array(list(itertools.combinations(range(N_POINTS), 3)), dtype=np.int32)

def _triangle_areas(points):
    """Compute all 286 triangle areas vectorized."""
    p1 = points[_TRI_IDX[:, 0]]
    p2 = points[_TRI_IDX[:, 1]]
    p3 = points[_TRI_IDX[:, 2]]
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)

def compute_score(points):
    """Compute the Heilbronn score: A_min / A_hull."""
    try:
        hull = ConvexHull(points)
        A_hull = hull.volume
    except Exception:
        return 0.0
    if A_hull <= 0:
        return 0.0
    areas = _triangle_areas(points)
    return np.min(areas) / A_hull

def _soft_min_score(points, beta=50.0):
    """Smooth surrogate: -log-sum-exp(-beta * areas) / A_hull. Higher is better."""
    try:
        hull = ConvexHull(points)
        A_hull = hull.volume
    except Exception:
        return -1e10
    if A_hull <= 0:
        return -1e10
    areas = _triangle_areas(points)
    # Soft minimum: approximates min(areas) smoothly
    log_sum_exp = np.log(np.sum(np.exp(-beta * areas)))
    soft_min = -log_sum_exp / beta
    return soft_min / A_hull

def _nelder_mead_optimize(points0, time_limit, beta=50.0):
    """Optimize using Nelder-Mead on the soft-min surrogate, then polish with hard min."""
    start = time.time()
    x0 = points0.flatten().copy()
    
    def objective(x):
        if time.time() - start > time_limit:
            return 1e10
        pts = x.reshape(N_POINTS, 2)
        return -_soft_min_score(pts, beta=beta)
    
    # Phase 1: Nelder-Mead on soft surrogate
    try:
        result = minimize(objective, x0, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-8,
                                   'adaptive': True})
        best_x = result.x
        best_pts = best_x.reshape(N_POINTS, 2)
        best_score = compute_score(best_pts)
    except Exception:
        best_pts = points0.copy()
        best_score = compute_score(best_pts)
    
    # Phase 2: Polish with hard min using smaller Nelder-Mead
    remaining = time_limit - (time.time() - start)
    if remaining > 5:
        def objective_hard(x):
            if time.time() - start > time_limit:
                return 1e10
            pts = x.reshape(N_POINTS, 2)
            s = compute_score(pts)
            return -s if np.isfinite(s) else 1e10
        
        try:
            result2 = minimize(objective_hard, best_x, method='Nelder-Mead',
                               options={'maxiter': 3000, 'xatol': 1e-7, 'fatol': 1e-10,
                                        'adaptive': True})
            pts2 = result2.x.reshape(N_POINTS, 2)
            score2 = compute_score(pts2)
            if score2 > best_score:
                best_pts = pts2
                best_score = score2
        except Exception:
            pass
    
    return best_pts, best_score

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the minimum triangle area ratio.
    Uses Nelder-Mead optimization with soft-min surrogate for efficient global search.
    """
    start_time = time.time()
    deadline = start_time + 590.0  # safety margin before 630s limit
    
    def time_left():
        return deadline - time.time()
    
    # Initial configurations
    def make_config_regular13():
        angles = np.linspace(0, 2 * np.pi, 13, endpoint=False)
        return np.column_stack([np.cos(angles), np.sin(angles)])
    
    def make_config_ring12_1():
        angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        outer = np.column_stack([np.cos(angles), np.sin(angles)])
        center = np.array([[0.0, 0.0]])
        return np.vstack([outer, center])
    
    def make_config_two_rings():
        angles1 = np.linspace(0, 2 * np.pi, 7, endpoint=False)
        angles2 = np.linspace(0, 2 * np.pi, 6, endpoint=False) + np.pi / 6
        outer = np.column_stack([np.cos(angles1), np.sin(angles1)])
        inner = np.column_stack([0.5 * np.cos(angles2), 0.5 * np.sin(angles2)])
        return np.vstack([outer, inner])
    
    def make_config_three_rings():
        a1 = np.linspace(0, 2 * np.pi, 5, endpoint=False)
        a2 = np.linspace(0, 2 * np.pi, 4, endpoint=False) + np.pi / 4
        a3 = np.linspace(0, 2 * np.pi, 4, endpoint=False)
        p1 = np.column_stack([np.cos(a1), np.sin(a1)])
        p2 = np.column_stack([0.55 * np.cos(a2), 0.55 * np.sin(a2)])
        p3 = np.column_stack([0.25 * np.cos(a3), 0.25 * np.sin(a3)])
        return np.vstack([p1, p2, p3])
    
    def make_config_center_rings():
        a1 = np.linspace(0, 2 * np.pi, 6, endpoint=False)
        a2 = np.linspace(0, 2 * np.pi, 6, endpoint=False) + np.pi / 6
        p1 = np.array([[0.0, 0.0]])
        p2 = np.column_stack([0.45 * np.cos(a1), 0.45 * np.sin(a1)])
        p3 = np.column_stack([np.cos(a2), np.sin(a2)])
        return np.vstack([p1, p2, p3])
    
    initial_configs = [
        make_config_regular13(),
        make_config_ring12_1(),
        make_config_two_rings(),
        make_config_three_rings(),
        make_config_center_rings(),
    ]
    
    # Add perturbed versions for diversity
    rng = np.random.default_rng(42)
    for base in initial_configs[:3]:
        perturbed = base + rng.normal(0, 0.05, size=base.shape)
        initial_configs.append(perturbed)
    
    best_points = None
    best_score = 0.0
    
    for i, config in enumerate(initial_configs):
        if time_left() < 30:
            break
        
        # Evaluate initial score
        score = compute_score(config)
        if score > best_score:
            best_score = score
            best_points = config.copy()
        
        # Allocate time per config based on remaining budget
        configs_remaining = len(initial_configs) - i
        time_per_config = min(50.0, (time_left() - 10) / max(configs_remaining, 1))
        
        if time_per_config < 10:
            continue
        
        # Optimize this config
        opt_pts, opt_score = _nelder_mead_optimize(config, time_per_config)
        
        if opt_score > best_score:
            best_score = opt_score
            best_points = opt_pts.copy()
    
    # Final polish on the best result
    if time_left() > 20 and best_points is not None:
        remaining = min(30.0, time_left() - 10)
        opt_pts, opt_score = _nelder_mead_optimize(best_points, remaining, beta=100.0)
        if opt_score > best_score:
            best_score = opt_score
            best_points = opt_pts.copy()
    
    # Ensure validity
    if best_points is None or not np.all(np.isfinite(best_points)):
        best_points = make_config_regular13()
    
    # Final validation
    try:
        hull = ConvexHull(best_points)
    except Exception:
        best_points = make_config_regular13()
    
    return best_points

# EVOLVE-BLOCK-END
