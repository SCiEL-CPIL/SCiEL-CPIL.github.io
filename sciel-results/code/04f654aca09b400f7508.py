"""Heilbronn convex regions: place 13 points to maximize min triangle area / hull area."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, differential_evolution
import time
import itertools

DEADLINE = 120.0  # seconds, with safety margin


def _hull_area(pts):
    """Compute convex hull area of points."""
    try:
        hull = ConvexHull(pts)
        return hull.volume  # In 2D, volume is area
    except Exception:
        return 0.0


def _min_triangle_area(pts):
    """Compute minimum triangle area among all triples."""
    n = len(pts)
    min_area = float('inf')
    for i, j, k in itertools.combinations(range(n), 3):
        a = pts[i]
        b = pts[j]
        c = pts[k]
        # Area = 0.5 * |cross(b-a, c-a)|
        cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        area = 0.5 * abs(cross)
        if area < min_area:
            min_area = area
    return min_area


def _score(pts):
    """Score: min triangle area / convex hull area."""
    n = len(pts)
    if n < 3:
        return 0.0
    ha = _hull_area(pts)
    if ha <= 1e-12:
        return 0.0
    mta = _min_triangle_area(pts)
    return mta / ha


def _smooth_min_triangle(pts, beta):
    """
    Smooth surrogate for min triangle area using soft-min via log-sum-exp.
    We want to maximize min area, so we minimize -min area.
    Using soft-min: -log(sum(exp(-beta * areas))) / beta approximates min(areas).
    """
    n = len(pts)
    areas = []
    for i, j, k in itertools.combinations(range(n), 3):
        a = pts[i]
        b = pts[j]
        c = pts[k]
        cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        area = 0.5 * abs(cross)
        areas.append(area)
    areas = np.array(areas)
    # Soft-min: approx min(areas) for large beta
    # To avoid overflow, subtract max
    max_a = np.max(areas)
    log_sum = np.log(np.sum(np.exp(-beta * (areas - max_a)))) + (-beta * max_a)
    soft_min = max_a - np.log(np.sum(np.exp(-beta * (areas - max_a)))) / beta
    return soft_min


def _smooth_score(pts, beta):
    """Smooth version of score for optimization."""
    n = len(pts)
    ha = _hull_area(pts)
    if ha <= 1e-12:
        return -1e6
    mta_smooth = _smooth_min_triangle(pts, beta)
    return mta_smooth / ha


def _initial_symmetric():
    """
    Create an initial symmetric configuration.
    Use a combination of:
    - Center point
    - Points on concentric regular polygons
    """
    pts = []
    # Center
    pts.append([0.5, 0.5])
    
    # Layer 1: 4 points on a square (radius r1)
    r1 = 0.25
    for angle in [np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4]:
        pts.append([0.5 + r1 * np.cos(angle), 0.5 + r1 * np.sin(angle)])
    
    # Layer 2: 4 points on a square rotated 45 degrees (radius r2)
    r2 = 0.35
    for angle in [0, np.pi/2, np.pi, 3*np.pi/2]:
        pts.append([0.5 + r2 * np.cos(angle), 0.5 + r2 * np.sin(angle)])
    
    # Layer 3: 3 points on a triangle (radius r3)
    r3 = 0.42
    for angle in [np.pi/2, np.pi/2 + 2*np.pi/3, np.pi/2 + 4*np.pi/3]:
        pts.append([0.5 + r3 * np.cos(angle), 0.5 + r3 * np.sin(angle)])
    
    return np.array(pts[:N])


def _refine_from_init(init_pts, time_limit):
    """
    Multi-stage refinement:
    1. Use smooth surrogate with increasing beta
    2. Final polish with exact objective
    """
    start_time = time.time()
    best_pts = init_pts.copy()
    best_score = _score(best_pts)
    
    # Stage 1: Smooth optimization with increasing beta
    betas = [5, 10, 20, 50, 100, 200]
    
    for beta in betas:
        if time.time() - start_time > time_limit * 0.7:
            break
        
        def objective(x):
            pts = x.reshape(N, 2)
            # Keep points in [0,1]^2 by penalty
            out_of_bounds = np.sum(np.maximum(0, pts - 1.0)) + np.sum(np.maximum(0, -pts))
            if out_of_bounds > 1e-6:
                return -_smooth_score(pts, beta) - 100 * out_of_bounds
            return -_smooth_score(pts, beta)
        
        try:
            result = minimize(
                objective,
                init_pts.flatten(),
                method='L-BFGS-B',
                bounds=[(0, 1)] * (2 * N),
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            new_pts = result.x.reshape(N, 2)
            new_score = _score(new_pts)
            if new_score > best_score:
                best_score = new_score
                best_pts = new_pts
                init_pts = new_pts
        except Exception:
            pass
    
    # Stage 2: Final polish with exact objective
    if time.time() - start_time < time_limit * 0.85:
        def objective_exact(x):
            pts = x.reshape(N, 2)
            out_of_bounds = np.sum(np.maximum(0, pts - 1.0)) + np.sum(np.maximum(0, -pts))
            if out_of_bounds > 1e-6:
                return -_score(pts) - 100 * out_of_bounds
            return -_score(pts)
        
        try:
            result = minimize(
                objective_exact,
                best_pts.flatten(),
                method='L-BFGS-B',
                bounds=[(0, 1)] * (2 * N),
                options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            new_pts = result.x.reshape(N, 2)
            new_score = _score(new_pts)
            if new_score > best_score:
                best_score = new_score
                best_pts = new_pts
        except Exception:
            pass
    
    return best_pts, best_score


def _differential_evolution_search(time_limit):
    """
    Use differential evolution for global search.
    """
    start_time = time.time()
    
    def objective(x):
        pts = x.reshape(N, 2)
        return -_score(pts)
    
    bounds = [(0, 1)] * (2 * N)
    
    try:
        result = differential_evolution(
            objective,
            bounds,
            maxiter=200,
            popsize=15,
            tol=1e-8,
            seed=42,
            polish=False
        )
        best_pts = result.x.reshape(N, 2)
        best_score = _score(best_pts)
    except Exception:
        best_pts = _initial_symmetric()
        best_score = _score(best_pts)
    
    return best_pts, best_score


def construct():
    """
    Main entry point: returns array of shape (N, 2).
    """
    start_time = time.time()
    deadline = start_time + DEADLINE - 5  # 5 second safety margin
    
    # Start with symmetric initialization
    init_pts = _initial_symmetric()
    best_pts, best_score = _refine_from_init(init_pts, time_limit=30)
    
    # If we have time, try differential evolution from multiple starts
    remaining = deadline - time.time()
    if remaining > 20:
        de_pts, de_score = _differential_evolution_search(time_limit=remaining * 0.8)
        if de_score > best_score:
            best_pts = de_pts
            best_score = de_score
    
    # Final polish
    remaining = deadline - time.time()
    if remaining > 5:
        best_pts, best_score = _refine_from_init(best_pts, time_limit=remaining * 0.8)
    
    # Ensure points are within [0,1]^2
    best_pts = np.clip(best_pts, 0, 1)
    
    return best_pts
# EVOLVE-BLOCK-END
