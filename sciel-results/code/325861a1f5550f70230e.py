"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from itertools import combinations

_DEADLINE = 115.0  # safety margin before 120s limit


def _min_triangle_ratio(pts):
    """Compute min triangle area / convex hull area for a set of 2D points."""
    pts = np.asarray(pts, dtype=np.float64)
    n = len(pts)
    if n < 3:
        return 0.0
    
    # Compute convex hull area using simple method
    # Find convex hull using gift wrapping or use scipy
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
    except Exception:
        return 0.0
    
    if hull_area <= 0:
        return 0.0
    
    # Find minimum triangle area
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = pts[i], pts[j], pts[k]
                area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
                if area < min_area:
                    min_area = area
    
    return min_area / hull_area


def _negative_ratio(pts):
    """Negative of the ratio for minimization."""
    return -_min_triangle_ratio(pts)


def construct():
    start_time = time.time()
    
    # Parameterize 14 points as centrally symmetric: 7 pairs (p_i, -p_i)
    # Each p_i is defined by radius r_i and angle theta_i
    # We optimize 14 parameters: 7 radii and 7 angles
    
    def _params_to_points(params):
        radii = params[:7]
        angles = params[7:14]
        pts = np.zeros((14, 2))
        for i in range(7):
            pts[i] = [radii[i] * np.cos(angles[i]), radii[i] * np.sin(angles[i])]
            pts[i + 7] = -pts[i]
        return pts
    
    def _objective(params):
        pts = _params_to_points(params)
        return _negative_ratio(pts)
    
    # Initial guess: regular 14-gon (centrally symmetric)
    best_params = None
    best_score = 0.0
    
    # Try multiple initializations
    n_trials = 8
    for trial in range(n_trials):
        if time.time() - start_time > _DEADLINE - 5:
            break
        
        if trial == 0:
            # Regular 14-gon
            angles = np.linspace(0, 2 * np.pi, 7, endpoint=False) + np.pi / 14
            radii = np.ones(7)
        else:
            # Perturbed regular 14-gon
            angles = np.linspace(0, 2 * np.pi, 7, endpoint=False) + np.pi / 14
            angles += 0.1 * np.random.randn(7)
            radii = 1.0 + 0.1 * np.random.randn(7)
            radii = np.abs(radii) + 0.1
        
        params0 = np.concatenate([radii, angles])
        
        # Optimize
        try:
            result = minimize(
                _objective,
                params0,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
            )
            score = -result.fun
            if score > best_score:
                best_score = score
                best_params = result.x.copy()
        except Exception:
            pass
    
    # Final polish with tighter tolerance if time permits
    if best_params is not None and time.time() - start_time < _DEADLINE - 10:
        try:
            result = minimize(
                _objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            score = -result.fun
            if score > best_score:
                best_score = score
                best_params = result.x.copy()
        except Exception:
            pass
    
    if best_params is None:
        # Fallback: regular 14-gon
        angles = np.linspace(0, 2 * np.pi, 14, endpoint=False)
        pts = np.column_stack([np.cos(angles), np.sin(angles)])
    else:
        pts = _params_to_points(best_params)
    
    # Normalize to unit area (convex hull area)
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
        if hull_area > 0:
            pts = pts / np.sqrt(hull_area)
    except Exception:
        pass
    
    return pts
# EVOLVE-BLOCK-END
