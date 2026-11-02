"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution
import time


def _score_points(pts):
    """Compute min triangle area / convex hull area."""
    n = len(pts)
    if n < 3:
        return 0.0
    
    # Compute convex hull area
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume  # In 2D, volume is area
        if hull_area <= 0:
            return 0.0
    except Exception:
        return 0.0
    
    # Compute all triangle areas using cross product
    # For points i, j, k: area = 0.5 * |cross(pj-pi, pk-pi)|
    min_area = float('inf')
    
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                # Cross product magnitude
                dx1 = pts[j, 0] - pts[i, 0]
                dy1 = pts[j, 1] - pts[i, 1]
                dx2 = pts[k, 0] - pts[i, 0]
                dy2 = pts[k, 1] - pts[i, 1]
                cross = abs(dx1 * dy2 - dy1 * dx2)
                area = 0.5 * cross
                if area < min_area:
                    min_area = area
    
    return min_area / hull_area


def _neg_score(params):
    """Negative score for minimization."""
    pts = params.reshape(-1, 2)
    return -_score_points(pts)


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    # Initial random points
    rng = np.random.RandomState(N)
    best_pts = rng.rand(N, 2)
    best_score = _score_points(best_pts)
    
    # Use differential evolution to maximize score
    # Parameters: 2N coordinates in [0, 1]
    bounds = [(0.0, 1.0)] * (2 * N)
    
    try:
        result = differential_evolution(
            _neg_score,
            bounds=bounds,
            seed=N,
            maxiter=50,
            popsize=15,
            tol=1e-8,
            callback=lambda x, convergence: None
        )
        
        if result.success and result.fun is not None:
            candidate_pts = result.x.reshape(-1, 2)
            candidate_score = _score_points(candidate_pts)
            if candidate_score > best_score:
                best_pts = candidate_pts
                best_score = candidate_score
        
        # Check time
        if time.time() > deadline - 5.0:
            return best_pts
        
    except Exception:
        pass
    
    # If we have time, try a local refinement
    if time.time() < deadline - 10.0:
        from scipy.optimize import minimize
        
        # Try multiple random restarts with local optimization
        for restart in range(5):
            if time.time() > deadline - 5.0:
                break
            
            init_pts = rng.rand(N, 2)
            init_params = init_pts.flatten()
            
            try:
                res = minimize(
                    _neg_score,
                    init_params,
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8}
                )
                
                candidate_pts = res.x.reshape(-1, 2)
                candidate_score = _score_points(candidate_pts)
                if candidate_score > best_score:
                    best_pts = candidate_pts
                    best_score = candidate_score
                    
            except Exception:
                continue
    
    return best_pts
# EVOLVE-BLOCK-END
