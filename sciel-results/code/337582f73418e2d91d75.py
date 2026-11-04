import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 14
TRIPLE_INDICES = np.array([
    (i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)
])

def _calculate_score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    points: (N, 2) array
    """
    if points.shape != (N, 2):
        return 0.0
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-9:
            return 0.0
    except Exception:
        return 0.0

    p_i = points[TRIPLE_INDICES[:, 0]]
    p_j = points[TRIPLE_INDICES[:, 1]]
    p_k = points[TRIPLE_INDICES[:, 2]]

    cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - \
            (p_j[:, 1] - p_i[:, 1]) * (p_k[:, 0] - p_i[:, 0])

    min_area = 0.5 * np.min(np.abs(cross))

    return min_area / hull_area

def _soft_min_obj(flat_points, beta):
    """
    Objective function for minimization: - (soft_min_triangle_area / hull_area)
    """
    pts = flat_points.reshape(N, 2)
    
    p_i = pts[TRIPLE_INDICES[:, 0]]
    p_j = pts[TRIPLE_INDICES[:, 1]]
    p_k = pts[TRIPLE_INDICES[:, 2]]

    cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - \
            (p_j[:, 1] - p_i[:, 1]) * (p_k[:, 0] - p_i[:, 0])
    
    areas = 0.5 * np.abs(cross)
    
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
        if hull_area <= 1e-9:
            return 1e6
    except Exception:
        return 1e6

    # Soft min: -1/beta * log(sum(exp(-beta * areas)))
    # Numerically stable:
    min_a = np.min(areas)
    shifted = areas - min_a
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    soft_min_val = min_a - (1.0/beta) * log_sum
    
    score = soft_min_val / hull_area
    return -score

def _polish(points, time_limit):
    """
    Polish points using L-BFGS-B with beta annealing.
    """
    best_pts = points.copy()
    best_score = _calculate_score(best_pts)
    start_time = time.time()
    
    # Beta annealing schedule
    betas = [10, 30, 100, 300]
    
    for beta in betas:
        if time.time() - start_time > time_limit - 1.0:
            break
        
        # Define objective with fixed beta
        def obj(x):
            return _soft_min_obj(x, beta)
        
        try:
            res = minimize(
                obj,
                best_pts.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'disp': False}
            )
            new_pts = res.x.reshape(N, 2)
            new_score = _calculate_score(new_pts)
            
            if new_score > best_score:
                best_score = new_score
                best_pts = new_pts
        except Exception:
            pass
            
    return best_score, best_pts

def _generate_regular_polygon(r=1.0, noise=0.0):
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    if noise > 0:
        rng = np.random.default_rng(42)
        angles += rng.normal(0, noise, N)
        radii = 1.0 + rng.normal(0, noise/2, N)
    else:
        radii = np.ones(N) * r
    
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.column_stack((x, y))

def _generate_random_points(seed):
    rng = np.random.default_rng(seed)
    r = np.sqrt(rng.random(N))
    theta = rng.random(N) * 2 * np.pi
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack((x, y))

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_score = 0.0
    best_points = None

    def try_update(score, points):
        nonlocal best_score, best_points
        if score > best_score:
            best_score = score
            best_points = points.copy()

    # Initial candidates
    candidates = []
    
    # 1. Regular polygon
    candidates.append(_generate_regular_polygon(1.0, noise=0.0))
    
    # 2. Jittered regular polygons
    for noise in [0.05, 0.1, 0.2]:
        candidates.append(_generate_regular_polygon(1.0, noise=noise))
        
    # 3. Random points
    for seed in range(5):
        candidates.append(_generate_random_points(seed))

    # Polish each candidate
    for i, cand in enumerate(candidates):
        if time.time() > deadline - 10:
            break
        
        # Allocate time based on remaining time and number of candidates
        # Simple heuristic: give each candidate equal share of remaining time
        remaining_cands = len(candidates) - i
        time_per_cand = min(10.0, (deadline - time.time()) / remaining_cands)
        
        score, pts = _polish(cand, time_per_cand)
        try_update(score, pts)
        
    # If we have time left, do a few more random restarts with polish
    if best_points is not None:
        # Use the best points as a starting point for further refinement?
        # Or just return.
        pass
    else:
        # Fallback
        best_points = _generate_regular_polygon(1.0)
        best_score = _calculate_score(best_points)

    return best_points
