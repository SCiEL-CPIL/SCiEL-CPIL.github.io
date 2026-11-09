import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def _triangle_areas(points):
    n = len(points)
    indices = np.array([(i, j, k) for i in range(n) for j in range(i+1, n) for k in range(j+1, n)])
    p1 = points[indices[:, 0]]
    p2 = points[indices[:, 1]]
    p3 = points[indices[:, 2]]
    areas = 0.5 * np.abs(
        p1[:, 0] * (p2[:, 1] - p3[:, 1]) +
        p2[:, 0] * (p3[:, 1] - p1[:, 1]) +
        p3[:, 0] * (p1[:, 1] - p2[:, 1])
    )
    return areas

def _score(points):
    try:
        if np.any(~np.isfinite(points)):
            return 0.0
        if len(points) != 13 or points.shape != (13, 2):
            return 0.0
        hull = ConvexHull(points)
        a_hull = hull.volume
        if a_hull <= 0:
            return 0.0
        areas = _triangle_areas(points)
        a_min = np.min(areas)
        if a_min <= 0:
            return 0.0
        return a_min / a_hull
    except Exception:
        return 0.0

def _generate_structured_points(r2, theta):
    r1 = 1.0
    pts = np.zeros((13, 2))
    pts[0] = [0, 0]
    for i in range(6):
        angle = i * np.pi / 3.0
        pts[i+1] = [r1 * np.cos(angle), r1 * np.sin(angle)]
    for i in range(6):
        angle = i * np.pi / 3.0 + theta
        pts[i+7] = [r2 * np.cos(angle), r2 * np.sin(angle)]
    return pts

def _generate_regular_polygon(n):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack((np.cos(angles), np.sin(angles)))

def heilbronn_convex13() -> np.ndarray:
    deadline = time.time() + 590  # Safety margin
    
    # Initial candidates
    candidates = []
    candidates.append(_generate_regular_polygon(13))
    candidates.append(_generate_structured_points(2.0, np.pi / 6.0))
    candidates.append(_generate_structured_points(1.5, np.pi / 12.0))
    
    # Add some random candidates
    rng = np.random.default_rng(42)
    for _ in range(10):
        pts = rng.uniform(-2, 2, (13, 2))
        candidates.append(pts)
    
    best_pts = candidates[0]
    best_score = _score(best_pts)
    for c in candidates[1:]:
        s = _score(c)
        if s > best_score:
            best_score = s
            best_pts = c

    # Multi-start Nelder-Mead
    # We have ~580 seconds.
    # Let's run 10 NM optimizations, each with a time limit.
    # Or run NM with maxiter set high and interrupt if time runs out?
    # scipy minimize doesn't have a time limit.
    # We can estimate iterations.
    # NM eval is fast (~0.5ms). 1000 iters ~ 0.5s.
    # We can do ~5000 iters per run easily.
    # 10 runs * 5000 iters = 50,000 evals.
    # 50,000 * 0.5ms = 25s.
    # This is very fast. We can do more runs or more iters.
    # Let's do 50 runs with 2000 iters each.
    # 50 * 2000 = 100,000 evals. ~50s.
    # We have 580s. We can do 500 runs?
    # 500 * 2000 = 1,000,000 evals. ~500s.
    # This fits perfectly.
    
    def _obj(x):
        pts = x.reshape(13, 2)
        return -_score(pts)
    
    # Generate more diverse starts
    # We can use the best points found so far as a start for some runs.
    # And random perturbations of the best points.
    
    starts = []
    for c in candidates:
        starts.append(c)
        # Add perturbed versions
        for _ in range(4):
            noise = rng.normal(0, 0.1, (13, 2))
            starts.append(c + noise)
            
    # Shuffle starts
    rng.shuffle(starts)
    
    # Run NM on each start
    # Time check inside loop
    for start in starts:
        if time.time() > deadline - 10:
            break
            
        try:
            res = minimize(_obj, start.flatten(), method='Nelder-Mead', 
                           options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-8})
            if res.fun < -best_score:
                pts_opt = res.x.reshape(13, 2)
                score_opt = _score(pts_opt)
                if score_opt > best_score:
                    best_score = score_opt
                    best_pts = pts_opt
        except Exception:
            pass

    if _score(best_pts) <= 0:
        best_pts = _generate_regular_polygon(13)

    return best_pts
