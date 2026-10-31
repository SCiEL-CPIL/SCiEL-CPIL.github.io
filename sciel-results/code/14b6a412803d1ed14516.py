import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time
from itertools import combinations

N = 13

def _min_triangle_area(pts):
    """Compute minimum triangle area over all triples (vectorized)."""
    n = len(pts)
    # Generate all combinations of 3 indices
    idx = np.array(list(combinations(range(n), 3)))
    # Extract coordinates
    p0 = pts[idx[:, 0]]
    p1 = pts[idx[:, 1]]
    p2 = pts[idx[:, 2]]
    # Compute cross product magnitude / 2 for each triangle
    v1 = p1 - p0
    v2 = p2 - p0
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _convex_hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume  # In 2D, volume is area
    except Exception:
        return 0.0

def _score(pts):
    """Score = min_triangle_area / convex_hull_area."""
    hull_area = _convex_hull_area(pts)
    if hull_area < 1e-12:
        return 0.0
    min_area = _min_triangle_area(pts)
    return min_area / hull_area

def _neg_score(params):
    """Negative score for minimization."""
    pts = params.reshape(N, 2)
    return -_score(pts)

def _local_search(pts, max_iter=500, tol=1e-10):
    """Local search using L-BFGS-B on the negative score."""
    x0 = pts.flatten()
    bounds = [(-2.0, 2.0)] * (2 * N)
    
    result = minimize(
        _neg_score,
        x0,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-8}
    )
    return result.x.reshape(N, 2)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    # Start with regular 13-gon
    angles = 2 * np.pi * np.arange(N) / N
    best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    best_score = _score(best_points)
    
    # Phase 1: Multiple random restarts with local search
    rng = np.random.RandomState(42)
    n_restarts = 20
    
    for restart in range(n_restarts):
        if time.time() > deadline - 30:
            break
        
        # Generate random initial points within unit disk
        if restart == 0:
            # Use regular polygon
            angles = 2 * np.pi * np.arange(N) / N
            pts = np.column_stack([np.cos(angles), np.sin(angles)])
        else:
            # Random points, then project to convex position
            angles = np.sort(rng.uniform(0, 2 * np.pi, N))
            radii = rng.uniform(0.5, 1.0, N)
            pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        
        # Local search
        try:
            refined = _local_search(pts, max_iter=300)
            s = _score(refined)
            if s > best_score:
                best_score = s
                best_points = refined.copy()
        except Exception:
            pass
    
    # Phase 2: Fine-tuning with more iterations
    if time.time() < deadline - 10:
        try:
            refined = _local_search(best_points, max_iter=1000, tol=1e-12)
            s = _score(refined)
            if s > best_score:
                best_score = s
                best_points = refined.copy()
        except Exception:
            pass
    
    # Phase 3: Perturbation-based escape from local minima
    if time.time() < deadline - 5:
        for _ in range(10):
            if time.time() > deadline - 5:
                break
            # Small random perturbation
            perturbation = rng.normal(0, 0.01, (N, 2))
            perturbed = best_points + perturbation
            try:
                refined = _local_search(perturbed, max_iter=200)
                s = _score(refined)
                if s > best_score:
                    best_score = s
                    best_points = refined.copy()
            except Exception:
                pass
    
    return best_points
