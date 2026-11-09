import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull
import time

N = 14

def _triangle_areas(points):
    """Compute all C(n,3) triangle areas efficiently."""
    n = len(points)
    areas = np.empty((n, n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                ax, ay = points[i]
                bx, by = points[j]
                cx, cy = points[k]
                area = 0.5 * abs((bx - ax) * (cy - ay) - (by - ay) * (cx - ax))
                areas[i, j, k] = area
    return areas

def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by triples of points."""
    n = len(points)
    min_area = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                ax, ay = points[i]
                bx, by = points[j]
                cx, cy = points[k]
                area = 0.5 * abs((bx - ax) * (cy - ay) - (by - ay) * (cx - ax))
                if area < min_area:
                    min_area = area
    return min_area

def _convex_hull_area(points):
    """Compute the area of the convex hull."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """Compute the score: min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area <= 1e-12:
        return -np.inf
    min_tri = _min_triangle_area(points)
    return min_tri / hull_area

def _apply_affine(regular_points, params):
    """Apply affine transformation to regular polygon points.
    
    params: [a, b, c, d, rot] where the 2x2 matrix is:
    [[a, b],
     [c, d]]
    followed by rotation by rot.
    """
    a, b, c, d, rot = params
    # Rotation matrix
    cos_r = np.cos(rot)
    sin_r = np.sin(rot)
    R = np.array([[cos_r, -sin_r],
                  [sin_r, cos_r]])
    # Affine matrix
    A = np.array([[a, b],
                  [c, d]])
    # Combined transformation
    M = R @ A
    return regular_points @ M.T

def _negative_score(params, regular_points):
    """Negative score for minimization."""
    points = _apply_affine(regular_points, params)
    s = _score(points)
    if s == -np.inf or np.isnan(s):
        return 1e10
    return -s

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s
    
    # Regular 14-gon vertices
    angles = 2 * np.pi * np.arange(N) / N
    regular_points = np.column_stack((np.cos(angles), np.sin(angles)))
    
    # Baseline score for regular polygon
    best_points = regular_points.copy()
    best_score = _score(best_points)
    
    # Define bounds for affine parameters
    # a, d: scaling factors (positive, around 1)
    # b, c: shear factors
    # rot: rotation angle
    bounds = [
        (0.5, 2.0),   # a
        (-1.0, 1.0),  # b
        (-1.0, 1.0),  # c
        (0.5, 2.0),   # d
        (-np.pi, np.pi)  # rot
    ]
    
    # Differential evolution for global search
    try:
        result = differential_evolution(
            _negative_score,
            bounds,
            args=(regular_points,),
            seed=42,
            maxiter=200,
            tol=1e-8,
            popsize=20,
            polish=False,
            init='sobol'
        )
        
        if result.success or result.fun < 1e10:
            candidate_points = _apply_affine(regular_points, result.x)
            candidate_score = _score(candidate_points)
            if candidate_score > best_score:
                best_points = candidate_points
                best_score = candidate_score
        
        # Check time
        elapsed = time.time() - start_time
        if elapsed > deadline - 30:
            return best_points
        
        # Local polish with Nelder-Mead
        def _neg_score_local(params):
            pts = _apply_affine(regular_points, params)
            s = _score(pts)
            if s == -np.inf or np.isnan(s):
                return 1e10
            return -s
        
        best_params = result.x.copy()
        best_local_score = best_score
        
        # Try multiple local optimizations from different starting points
        local_starts = [best_params]
        # Add some perturbed starts
        rng = np.random.RandomState(42)
        for _ in range(5):
            perturbed = best_params + rng.normal(0, 0.1, size=5)
            perturbed[0] = np.clip(perturbed[0], 0.5, 2.0)
            perturbed[3] = np.clip(perturbed[3], 0.5, 2.0)
            perturbed[4] = np.clip(perturbed[4], -np.pi, np.pi)
            local_starts.append(perturbed)
        
        for ls in local_starts:
            elapsed = time.time() - start_time
            if elapsed > deadline - 20:
                break
            try:
                res = minimize(
                    _neg_score_local,
                    ls,
                    args=(regular_points,),
                    method='Nelder-Mead',
                    options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12}
                )
                if res.fun < 1e10:
                    cand_pts = _apply_affine(regular_points, res.x)
                    cand_score = _score(cand_pts)
                    if cand_score > best_local_score:
                        best_local_score = cand_score
                        best_params = res.x.copy()
            except Exception:
                pass
        
        if best_local_score > best_score:
            best_points = _apply_affine(regular_points, best_params)
            best_score = best_local_score
            
    except Exception:
        pass
    
    # Final check: ensure we have a valid point set
    if best_score <= 0 or np.isnan(best_score):
        # Fallback to regular polygon
        best_points = regular_points.copy()
    
    return best_points
