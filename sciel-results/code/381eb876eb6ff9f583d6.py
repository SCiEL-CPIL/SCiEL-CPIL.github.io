"""Heilbronn convex regions: 13 points maximizing min triangle area / hull area."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 13
DEADLINE_MARGIN = 10.0  # seconds to leave before the 120s wall-clock limit

def _min_triangle_area(points):
    """Compute the minimum area over all triples of points using vectorization."""
    n = len(points)
    if n < 3:
        return 0.0
    
    # Generate all combinations of indices (i, j, k) with i < j < k
    # For N=13, there are 286 combinations, which is small enough for a list
    idx = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                idx.append((i, j, k))
    idx = np.array(idx)
    
    p1 = points[idx[:, 0]]
    p2 = points[idx[:, 1]]
    p3 = points[idx[:, 2]]
    
    # Area = 0.5 * | (x2-x1)(y3-y1) - (y2-y1)(x3-x1) |
    ax = p2[:, 0] - p1[:, 0]
    ay = p2[:, 1] - p1[:, 1]
    bx = p3[:, 0] - p1[:, 0]
    by = p3[:, 1] - p1[:, 1]
    
    areas = 0.5 * np.abs(ax * by - ay * bx)
    return np.min(areas)

def _hull_area(points):
    """Compute the area of the convex hull of the points."""
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is area
    except Exception:
        return 0.0

def _score(points):
    """Compute the score: min triangle area / convex hull area."""
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_tri = _min_triangle_area(points)
    return min_tri / hull_area

def _negative_score(points):
    """Negative of the score for minimization."""
    return -_score(points)

def _regular_polygon(n, center=(0, 0), radius=1.0):
    """Generate a regular n-gon."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([
        center[0] + radius * np.cos(angles),
        center[1] + radius * np.sin(angles)
    ])

def _make_12gon_center():
    """Create a regular 12-gon with a center point."""
    angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    pts = np.column_stack([
        np.cos(angles),
        np.sin(angles)
    ])
    # Add center
    pts = np.vstack([pts, np.array([0.0, 0.0])])
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 120.0 - DEADLINE_MARGIN
    
    # Initial candidates
    # 1. Regular 13-gon
    init_13 = _regular_polygon(N, radius=1.0)
    # 2. Regular 12-gon + center
    init_12c = _make_12gon_center()
    
    best_points = init_13.copy()
    best_score = _score(best_points)
    
    # Also track the score of the 12-gon+center config
    score_12c = _score(init_12c)
    if score_12c > best_score:
        best_score = score_12c
        best_points = init_12c.copy()

    # Optimization loop with restarts and perturbations
    rng = np.random.RandomState(42)
    
    # List of starting points to try
    starts = [
        init_13,
        init_12c,
        init_13 + rng.normal(0, 0.05, (N, 2)),
        init_12c + rng.normal(0, 0.05, (N, 2)),
        # A more "spread out" random start
        rng.uniform(-1, 1, (N, 2))
    ]
    
    current_best_points = best_points.copy()
    current_best_score = best_score
    
    for i, start_pts in enumerate(starts):
        if time.time() > deadline:
            break
            
        # Evaluate the start point
        s = _score(start_pts)
        if s > current_best_score:
            current_best_score = s
            current_best_points = start_pts.copy()
            best_score = s
            best_points = start_pts.copy()
        
        # Run Nelder-Mead from this start
        # We use a wrapper to flatten/unflatten for minimize
        def objective(flat_pts):
            pts = flat_pts.reshape(N, 2)
            return -_score(pts)
            
        try:
            result = minimize(
                objective,
                start_pts.flatten(),
                method='Nelder-Mead',
                options={
                    'maxiter': 3000, 
                    'xatol': 1e-8, 
                    'fatol': 1e-10, 
                    'adaptive': True
                }
            )
            
            opt_pts = result.x.reshape(N, 2)
            opt_score = _score(opt_pts)
            
            if opt_score > current_best_score:
                current_best_score = opt_score
                current_best_points = opt_pts.copy()
                best_score = opt_score
                best_points = opt_pts.copy()
                
        except Exception:
            continue
            
        # If we have time, try a "kick" from the best local solution found so far
        # This helps escape local minima
        if time.time() < deadline - 5.0:
            # Perturb the current best
            kick_size = 0.1
            kicked_pts = current_best_points + rng.normal(0, kick_size, (N, 2))
            
            # Evaluate kick
            kick_score = _score(kicked_pts)
            if kick_score > current_best_score:
                current_best_score = kick_score
                current_best_points = kicked_pts.copy()
                best_score = kick_score
                best_points = kicked_pts.copy()
            else:
                # Try optimizing from the kick
                try:
                    result_kick = minimize(
                        objective,
                        kicked_pts.flatten(),
                        method='Nelder-Mead',
                        options={
                            'maxiter': 1000, 
                            'xatol': 1e-8, 
                            'fatol': 1e-10, 
                            'adaptive': True
                        }
                    )
                    kick_opt_pts = result_kick.x.reshape(N, 2)
                    kick_opt_score = _score(kick_opt_pts)
                    
                    if kick_opt_score > current_best_score:
                        current_best_score = kick_opt_score
                        current_best_points = kick_opt_pts.copy()
                        best_score = kick_opt_score
                        best_points = kick_opt_pts.copy()
                except Exception:
                    pass

    # Final polish: run a high-precision optimization from the global best
    if time.time() < deadline - 2.0:
        try:
            def objective_polish(flat_pts):
                pts = flat_pts.reshape(N, 2)
                return -_score(pts)
                
            result_final = minimize(
                objective_polish,
                best_points.flatten(),
                method='Nelder-Mead',
                options={
                    'maxiter': 5000, 
                    'xatol': 1e-10, 
                    'fatol': 1e-12, 
                    'adaptive': True
                }
            )
            
            final_pts = result_final.x.reshape(N, 2)
            final_score = _score(final_pts)
            
            if final_score > best_score:
                best_score = final_score
                best_points = final_pts.copy()
        except Exception:
            pass

    return best_points
