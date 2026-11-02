"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution

# Deadline guard
DEADLINE = 150.0
START_TIME = time.time()

def _time_left():
    return DEADLINE - (time.time() - START_TIME) - 5.0  # 5s safety margin


def _min_triangle_area(pts):
    """Compute the minimum area over all triples of points."""
    n = len(pts)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Area of triangle (i, j, k) = 0.5 * |cross(pj-pi, pk-pi)|
                ax = pts[j, 0] - pts[i, 0]
                ay = pts[j, 1] - pts[i, 1]
                bx = pts[k, 0] - pts[i, 0]
                by = pts[k, 1] - pts[i, 1]
                area = 0.5 * abs(ax * by - ay * bx)
                if area < min_area:
                    min_area = area
    return min_area


def _score_points(pts):
    """Compute min triangle area / convex hull area."""
    n = len(pts)
    if n < 3:
        return 0.0
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume  # In 2D, volume is area
        if hull_area <= 0:
            return 0.0
    except Exception:
        return 0.0
    min_area = _min_triangle_area(pts)
    return min_area / hull_area


def _neg_score(params):
    """Objective function for differential evolution (minimize negative score)."""
    if _time_left() < 10.0:
        return 0.0  # Return a neutral value if time is running out
    n = len(params)
    pts = np.array(params).reshape((-1, 2))
    # Check for degenerate configurations
    if np.any(pts < -10) or np.any(pts > 10):
        return 0.0
    score = _score_points(pts)
    return -score


def construct():
    """Construct 13 points to maximize the minimum triangle area over convex hull area."""
    global START_TIME
    START_TIME = time.time()
    
    best_score = 0.0
    best_params = None
    
    # Strategy 1: Symmetric regular polygon with inner points
    # Start with a regular 7-gon (hull) and 6 inner points arranged symmetrically
    def _regular_polygon(n, radius, center=(0, 0)):
        angles = 2 * np.pi * np.arange(n) / n - np.pi / 2
        return np.column_stack([center[0] + radius * np.cos(angles),
                                center[1] + radius * np.sin(angles)])
    
    # Try several symmetric configurations
    # Configuration: 7 points on outer circle, 6 points on inner circle
    for outer_r in [1.0, 1.2, 1.5]:
        for inner_r in [0.3, 0.5, 0.7]:
            for offset in [0, np.pi / 7, np.pi / 14]:
                outer_pts = _regular_polygon(7, outer_r)
                inner_angles = 2 * np.pi * np.arange(6) / 6 + offset
                inner_pts = np.column_stack([inner_r * np.cos(inner_angles),
                                             inner_r * np.sin(inner_angles)])
                pts = np.vstack([outer_pts, inner_pts])
                score = _score_points(pts)
                if score > best_score:
                    best_score = score
                    best_params = pts.flatten()
    
    # Configuration: 6 points on outer hexagon, 7 points on inner circle
    for outer_r in [1.0, 1.2]:
        for inner_r in [0.3, 0.5]:
            for offset in [0, np.pi / 6]:
                outer_pts = _regular_polygon(6, outer_r)
                inner_angles = 2 * np.pi * np.arange(7) / 7 + offset
                inner_pts = np.column_stack([inner_r * np.cos(inner_angles),
                                             inner_r * np.sin(inner_angles)])
                pts = np.vstack([outer_pts, inner_pts])
                score = _score_points(pts)
                if score > best_score:
                    best_score = score
                    best_params = pts.flatten()
    
    # Configuration: 5 points on outer pentagon, 8 points on two inner circles
    for outer_r in [1.0, 1.3]:
        for r1 in [0.2, 0.4]:
            for r2 in [0.5, 0.7]:
                outer_pts = _regular_polygon(5, outer_r)
                inner1_angles = 2 * np.pi * np.arange(4) / 4
                inner1_pts = np.column_stack([r1 * np.cos(inner1_angles),
                                              r1 * np.sin(inner1_angles)])
                inner2_angles = 2 * np.pi * np.arange(4) / 4 + np.pi / 4
                inner2_pts = np.column_stack([r2 * np.cos(inner2_angles),
                                              r2 * np.sin(inner2_angles)])
                pts = np.vstack([outer_pts, inner1_pts, inner2_pts])
                score = _score_points(pts)
                if score > best_score:
                    best_score = score
                    best_params = pts.flatten()
    
    # Strategy 2: Differential evolution for global optimization
    if _time_left() > 30.0:
        bounds = [(-2, 2)] * (2 * N)
        # Use the best symmetric configuration as initial guess
        if best_params is not None:
            x0 = best_params
        else:
            x0 = np.random.uniform(-1, 1, 2 * N)
        
        try:
            result = differential_evolution(
                _neg_score,
                bounds=bounds,
                seed=42,
                maxiter=200,
                tol=1e-6,
                popsize=15,
                init='sobol',
                polish=True
            )
            if result.fun < 0:
                score = -result.fun
                if score > best_score:
                    best_score = score
                    best_params = result.x
        except Exception:
            pass
    
    # Strategy 3: Local refinement using Nelder-Mead from best point
    if best_params is not None and _time_left() > 10.0:
        from scipy.optimize import minimize
        try:
            result = minimize(
                _neg_score,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            if result.fun < 0:
                score = -result.fun
                if score > best_score:
                    best_score = score
                    best_params = result.x
        except Exception:
            pass
    
    # Return the best configuration found
    if best_params is not None:
        pts = np.array(best_params).reshape((-1, 2))
        # Normalize to unit area convex hull
        try:
            hull = ConvexHull(pts)
            hull_area = hull.volume
            if hull_area > 0:
                # Scale so hull area = 1
                scale = 1.0 / np.sqrt(hull_area)
                pts = pts * scale
        except Exception:
            pass
        return pts
    else:
        # Fallback: regular 13-gon
        angles = 2 * np.pi * np.arange(N) / N
        return np.column_stack([np.cos(angles), np.sin(angles)])

# EVOLVE-BLOCK-END
