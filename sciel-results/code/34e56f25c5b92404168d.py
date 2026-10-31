"""Rewritten program for Heilbronn convex regions with soft-min annealing."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull


def _all_triangle_areas(points):
    """Vectorized computation of all C(n,3) triangle areas."""
    n = len(points)
    # Get all pairs
    i_idx, j_idx, k_idx = np.indices((n, n, n))
    i_idx = i_idx[i_idx < j_idx].ravel()
    j_idx = j_idx[i_idx < j_idx].ravel()
    k_idx = k_idx[i_idx < j_idx].ravel()
    # Filter for i < j < k
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_idx = i_idx[mask]
    j_idx = j_idx[mask]
    k_idx = k_idx[mask]
    
    p1 = points[i_idx]
    p2 = points[j_idx]
    p3 = points[k_idx]
    
    # Cross product magnitude / 2
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p3[:, 0] - p1[:, 0]) * (p2[:, 1] - p1[:, 1])
    areas = 0.5 * np.abs(cross)
    return areas


def _convex_hull_area(points):
    """Compute the area of the convex hull."""
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is area
    except Exception:
        return 0.0


def _score(points):
    """Compute the score: min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area <= 1e-12:
        return 0.0
    areas = _all_triangle_areas(points)
    min_area = np.min(areas)
    return min_area / hull_area


def _soft_min_score(points, beta):
    """Compute soft-min score for smoother optimization landscape."""
    hull_area = _convex_hull_area(points)
    if hull_area <= 1e-12:
        return 0.0
    areas = _all_triangle_areas(points)
    # Soft min using log-sum-exp trick
    min_area = np.min(areas)
    scaled = areas / max(min_area, 1e-15)
    # Use exp(-beta * (area - min_area) / min_area) weighted average
    weights = np.exp(-beta * (areas - min_area) / max(min_area, 1e-15))
    soft_min = min_area + np.sum(weights * (areas - min_area)) / np.sum(weights)
    return soft_min / hull_area


def _regular_polygon_points(n, radius=1.0, center=(0.0, 0.0)):
    """Generate n points on a regular polygon."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([
        center[0] + radius * np.cos(angles),
        center[1] + radius * np.sin(angles)
    ])


def _structured_initial(n, variant=0):
    """Generate structured initial configurations for n points."""
    if variant == 0:
        return _regular_polygon_points(n, radius=1.0)
    elif variant == 1:
        outer = _regular_polygon_points(7, radius=1.0)
        inner = _regular_polygon_points(6, radius=0.5)
        return np.vstack([outer, inner])
    elif variant == 2:
        outer = _regular_polygon_points(5, radius=1.0)
        mid = _regular_polygon_points(5, radius=0.6)
        inner = _regular_polygon_points(3, radius=0.25)
        return np.vstack([outer, mid, inner])
    elif variant == 3:
        pts = []
        pts.append([0.0, 0.0])
        for i in range(6):
            angle = i * np.pi / 3
            pts.append([0.5 * np.cos(angle), 0.5 * np.sin(angle)])
        for i in range(6):
            angle = (i + 0.5) * np.pi / 3
            pts.append([1.0 * np.cos(angle), 1.0 * np.sin(angle)])
        return np.array(pts)
    elif variant == 4:
        pts = _regular_polygon_points(n, radius=1.0)
        rng = np.random.RandomState(variant + 100)
        pts = pts + 0.05 * rng.randn(n, 2)
        return pts
    else:
        return _regular_polygon_points(n, radius=1.0)


def _anneal_optimize(points, deadline, start_beta=10.0, end_beta=1000.0, max_iter=5000):
    """Anneal optimization: start with soft-min (smooth) and increase beta to hard-min."""
    start_time = time.time()
    n = len(points)
    x = points.flatten().copy()
    
    best_x = x.copy()
    best_score = _score(points)
    
    # Number of annealing steps
    n_steps = 20
    betas = np.geomspace(start_beta, end_beta, n_steps)
    
    for beta in betas:
        if time.time() > deadline:
            break
        
        def objective(x_flat):
            pts = x_flat.reshape(-1, 2)
            return -_soft_min_score(pts, beta)
        
        try:
            result = minimize(
                objective,
                x,
                method='Nelder-Mead',
                options={'maxiter': max_iter // n_steps, 'xatol': 1e-9, 'fatol': 1e-12, 'adaptive': True}
            )
            x = result.x
            score = _score(x.reshape(-1, 2))
            if score > best_score:
                best_score = score
                best_x = x.copy()
        except Exception:
            continue
        
        if time.time() > deadline:
            break
    
    return best_x.reshape(-1, 2), best_score


def construct():
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    best_points = None
    best_score = 0.0
    
    # Try multiple initial configurations with annealing
    for variant in range(5):
        if time.time() > deadline:
            break
        
        initial = _structured_initial(N, variant=variant)
        
        try:
            optimized, score = _anneal_optimize(initial, deadline)
            if score > best_score:
                best_score = score
                best_points = optimized.copy()
        except Exception:
            continue
    
    # If no optimization succeeded, return the best initial configuration
    if best_points is None:
        best_points = _structured_initial(N, variant=0)
    
    return best_points
# EVOLVE-BLOCK-END
