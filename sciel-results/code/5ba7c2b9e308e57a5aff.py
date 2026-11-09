"""Heilbronn convex regions: 13 points maximizing min triangle area / hull area."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13
DEADLINE = 115.0  # safety margin below 120s limit


def _min_triangle_area(points):
    """Compute the minimum triangle area among all triples of points."""
    n = len(points)
    idx = np.array(list(combinations(range(n), 3)))
    p1 = points[idx[:, 0]]
    p2 = points[idx[:, 1]]
    p3 = points[idx[:, 2]]
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    areas = 0.5 * np.abs(cross)
    return np.min(areas)


def _hull_area(points):
    """Compute the convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        # Fallback: use shoelace formula on sorted points
        # This is a rough fallback
        return 1.0


def _score(points):
    """Score = min triangle area / convex hull area."""
    min_ta = _min_triangle_area(points)
    hull_a = _hull_area(points)
    if hull_a <= 1e-12:
        return 0.0
    return min_ta / hull_a


def _negative_score(points):
    """Negative score for minimization."""
    return -_score(points)


def _objective(params):
    """Objective function for optimizer: negative of score."""
    points = params.reshape(N, 2)
    return _negative_score(points)


def _regular_polygon(n, radius=1.0, center=(0.0, 0.0)):
    """Generate a regular n-gon centered at origin."""
    angles = 2 * np.pi * np.arange(n) / n
    pts = radius * np.column_stack([np.cos(angles), np.sin(angles)])
    pts[:, 0] += center[0]
    pts[:, 1] += center[1]
    return pts


def construct():
    start_time = time.time()
    
    best_score = -1.0
    best_points = None

    # Generate diverse initializations
    initializations = []
    
    # 1. Regular 13-gon
    initializations.append(_regular_polygon(13, radius=1.0).flatten())
    
    # 2. 12-gon + center
    outer_12 = _regular_polygon(12, radius=1.0)
    center_pt = np.array([[0.0, 0.0]])
    initializations.append(np.vstack([outer_12, center_pt]).flatten())
    
    # 3. Jittered 13-gon
    rng = np.random.RandomState(42)
    jittered = _regular_polygon(13, radius=1.0)
    jittered += rng.normal(0, 0.1, jittered.shape)
    initializations.append(jittered.flatten())
    
    # 4. Random points in unit disk
    for _ in range(3):
        angles = rng.uniform(0, 2*np.pi, N)
        radii = np.sqrt(rng.uniform(0, 1, N))
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        initializations.append(pts.flatten())
    
    # 5. Grid-like structure (3x3 grid + 4 extra points)
    grid_pts = []
    for i in range(3):
        for j in range(3):
            grid_pts.append([(i-1)*0.5, (j-1)*0.5])
    # Add 4 more points to make 13
    extra = [
        [0.0, 0.0],
        [0.5, 0.5],
        [-0.5, -0.5],
        [0.5, -0.5]
    ]
    grid_pts.extend(extra)
    initializations.append(np.array(grid_pts).flatten())
    
    # Optimize each initialization using L-BFGS-B
    for x0 in initializations:
        if time.time() - start_time > DEADLINE:
            break
            
        try:
            res = minimize(
                _objective,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-6}
            )
            
            candidate = res.x.reshape(N, 2)
            c_score = _score(candidate)
            
            if c_score > best_score:
                best_score = c_score
                best_points = candidate.copy()
                
        except Exception:
            pass
    
    # If no valid result found, fallback to regular 13-gon
    if best_points is None:
        best_points = _regular_polygon(13, radius=1.0)
    
    # Final polish with Nelder-Mead on the best result
    if time.time() - start_time < DEADLINE - 5:
        try:
            x0 = best_points.flatten()
            res_nm = minimize(
                _objective,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            candidate_nm = res_nm.x.reshape(N, 2)
            c_score_nm = _score(candidate_nm)
            
            if c_score_nm > best_score:
                best_score = c_score_nm
                best_points = candidate_nm.copy()
                
        except Exception:
            pass
    
    return best_points
