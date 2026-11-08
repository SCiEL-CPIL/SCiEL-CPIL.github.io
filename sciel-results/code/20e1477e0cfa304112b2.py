"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import itertools

def _triangle_areas(pts):
    """Compute areas of all triples of points."""
    n = len(pts)
    areas = []
    for i, j, k in itertools.combinations(range(n), 3):
        a = pts[i]
        b = pts[j]
        c = pts[k]
        # Area = 0.5 * |cross(b-a, c-a)|
        ab = b - a
        ac = c - a
        area = 0.5 * abs(ab[0] * ac[1] - ab[1] * ac[0])
        areas.append(area)
    return np.array(areas)

def _hull_area(pts):
    """Compute area of convex hull of points."""
    try:
        hull = ConvexHull(pts)
        return hull.volume  # In 2D, volume is area
    except Exception:
        # Fallback: if points are collinear or degenerate
        return 1e-10

def _score(pts):
    """Compute the score: min triangle area / hull area."""
    areas = _triangle_areas(pts)
    min_area = np.min(areas)
    hull = _hull_area(pts)
    if hull < 1e-10:
        return 0.0
    return min_area / hull

def _soft_min_objective(pts_flat, t=0.01):
    """
    Objective to minimize: -soft_min(triangle_areas) / hull_area
    We want to maximize min_area / hull_area, so minimize negative of that.
    Use soft-min for differentiability.
    """
    pts = pts_flat.reshape(N, 2)
    areas = _triangle_areas(pts)
    hull = _hull_area(pts)
    if hull < 1e-10:
        return 1e10
    # Soft min: -logsumexp(-areas / t) * t
    # This approximates min(areas) when t is small
    neg_areas = -areas / t
    max_neg = np.max(neg_areas)
    logsumexp = max_neg + np.log(np.sum(np.exp(neg_areas - max_neg)))
    soft_min = -logsumexp * t
    # We want to maximize soft_min / hull, so minimize -soft_min / hull
    return -soft_min / hull

def _regular_polygon(n, radius=1.0):
    """Generate a regular n-gon."""
    angles = 2 * np.pi * np.arange(n) / n
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def _random_points(n, seed):
    """Generate random points in a unit circle."""
    rng = np.random.RandomState(seed)
    r = rng.uniform(0, 1, n) ** 0.5
    theta = rng.uniform(0, 2 * np.pi, n)
    return np.column_stack([r * np.cos(theta), r * np.sin(theta)])

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin: 10s before 150s limit
    
    best_score = -np.inf
    best_pts = None
    
    # Initial configurations to try
    configs = []
    
    # Regular 14-gon
    configs.append(_regular_polygon(N, radius=1.0))
    
    # Regular 14-gon with different radius
    configs.append(_regular_polygon(N, radius=1.5))
    
    # Random points with different seeds
    for seed in [42, 123, 456, 789, 1000, 2000, 3000, 4000]:
        if time.time() > deadline - 30:
            break
        configs.append(_random_points(N, seed))
    
    # Try each configuration
    for idx, init_pts in enumerate(configs):
        if time.time() > deadline - 20:
            break
        
        # Evaluate initial score
        init_score = _score(init_pts)
        if init_score > best_score:
            best_score = init_score
            best_pts = init_pts.copy()
        
        # Optimize using L-BFGS-B
        try:
            result = minimize(
                _soft_min_objective,
                init_pts.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
            )
            
            opt_pts = result.x.reshape(N, 2)
            opt_score = _score(opt_pts)
            
            if opt_score > best_score:
                best_score = opt_score
                best_pts = opt_pts.copy()
                
        except Exception as e:
            pass
        
        # Check time
        if time.time() > deadline - 10:
            break
    
    # Final polish: if we have time, do a more precise optimization
    if best_pts is not None and time.time() < deadline - 15:
        try:
            # Use a smaller t for more accurate min approximation
            def _final_objective(pts_flat):
                pts = pts_flat.reshape(N, 2)
                areas = _triangle_areas(pts)
                hull = _hull_area(pts)
                if hull < 1e-10:
                    return 1e10
                t = 0.005
                neg_areas = -areas / t
                max_neg = np.max(neg_areas)
                logsumexp = max_neg + np.log(np.sum(np.exp(neg_areas - max_neg)))
                soft_min = -logsumexp * t
                return -soft_min / hull
            
            result = minimize(
                _final_objective,
                best_pts.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
            )
            
            polish_pts = result.x.reshape(N, 2)
            polish_score = _score(polish_pts)
            
            if polish_score > best_score:
                best_score = polish_score
                best_pts = polish_pts.copy()
                
        except Exception as e:
            pass
    
    # Ensure we return a valid result
    if best_pts is None:
        # Fallback: regular polygon
        best_pts = _regular_polygon(N, radius=1.0)
    
    return best_pts
# EVOLVE-BLOCK-END
