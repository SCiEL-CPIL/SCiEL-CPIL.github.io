import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution, minimize

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
    deadline = time.time() + 580
    
    # Candidate 1: Structured
    pts_structured = _generate_structured_points(2.0, np.pi / 6.0)
    # Candidate 2: Regular 13-gon
    pts_regular = _generate_regular_polygon(13)
    
    best_pts = pts_structured
    best_score = _score(best_pts)
    if _score(pts_regular) > best_score:
        best_pts = pts_regular
        best_score = _score(best_pts)
    
    # Use Differential Evolution for global optimization
    # This is a structural change from local Nelder-Mead to global DE.
    # The objective function is highly non-smooth (min of 286 triangle areas / hull area).
    # DE is robust to non-smoothness.
    
    def _obj(x):
        pts = x.reshape(13, 2)
        return -_score(pts)
    
    # Bounds: normalize roughly to [-2, 2]
    # Since the problem is scale-invariant, the optimizer might drift, 
    # but the ratio is invariant. 
    # However, DE needs finite bounds. 
    # A large box is needed to allow the shape to change.
    bounds = [(-3, 3)] * 26
    
    # We need to be careful with the number of iterations and population size to fit in time.
    # 13 points * 2 coords = 26 dimensions.
    # DE with popsize=10 -> 10*26 = 260 evaluations per generation.
    # Each evaluation is ~1ms (triangle areas + hull).
    # 260ms per generation.
    # 580s / 0.26s ~ 2200 generations.
    # That's a lot. We can try maxiter=500 to be safe, or let it run until time.
    
    try:
        # maxiter is a soft limit if we use a callback, but DE doesn't have a time callback easily.
        # We can set a reasonable maxiter and rely on the fact that it might finish early if tol is met, 
        # but usually it runs full maxiter.
        # Let's estimate: 500 generations * 10 pop * 26 dims = 130,000 evaluations.
        # If each eval is 0.5ms, that's 65 seconds.
        # If 1ms, 130 seconds.
        # So 500 iterations is safe.
        
        # To improve quality, we can run multiple DE runs with different seeds? 
        # Or just one long run.
        # Let's try one run with maxiter=400 and popsize=20.
        # 400 * 20 * 26 = 208,000 evals. ~200-400s.
        
        res = differential_evolution(
            _obj, 
            bounds, 
            maxiter=400, 
            popsize=20, 
            tol=1e-8, 
            seed=42,
            init='latinhypercube'
        )
        
        if res.fun < -best_score:
            pts_opt = res.x.reshape(13, 2)
            score_opt = _score(pts_opt)
            if score_opt > best_score:
                best_score = score_opt
                best_pts = pts_opt
                
    except Exception:
        pass
    
    # Refinement with Nelder-Mead if time permits
    if time.time() < deadline - 30:
        try:
            res = minimize(_obj, best_pts.flatten(), method='Nelder-Mead', options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8})
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
