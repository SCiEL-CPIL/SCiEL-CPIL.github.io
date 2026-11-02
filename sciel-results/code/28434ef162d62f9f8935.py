"""Rewritten program for Heilbronn convex regions: maximize min triangle area / hull area for 13 points."""
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull

N = 13
DEADLINE = 140  # safety margin before 150s kill


def _score_points(pts):
    """Compute min triangle area / convex hull area for a set of 2D points."""
    n = len(pts)
    if n < 3:
        return 0.0
    
    # Ensure points are 2D array
    pts = np.asarray(pts, dtype=np.float64)
    
    # Compute convex hull area
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume  # In 2D, volume is area
        if hull_area <= 1e-15:
            return 0.0
    except Exception:
        return 0.0
    
    # Compute all triangle areas using cross product
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Cross product magnitude / 2 = triangle area
                cross = (pts[j, 0] - pts[i, 0]) * (pts[k, 1] - pts[i, 1]) - \
                        (pts[j, 1] - pts[i, 1]) * (pts[k, 0] - pts[i, 0])
                area = 0.5 * abs(cross)
                if area < min_area:
                    min_area = area
    
    return min_area / hull_area


def _neg_score(pts):
    """Negative score for minimization."""
    return -_score_points(pts)


def _refine_with_nelder_mead(pts, time_budget):
    """Refine a point configuration using Nelder-Mead."""
    start = time.time()
    best_pts = pts.copy()
    best_score = _score_points(best_pts)
    
    # Use Nelder-Mead on the flattened parameter vector
    def objective(x):
        p = x.reshape(N, 2)
        return -_score_points(p)
    
    try:
        result = minimize(
            objective,
            best_pts.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
        )
        if result.fun < -best_score:
            refined = result.x.reshape(N, 2)
            refined_score = _score_points(refined)
            if refined_score > best_score:
                best_pts = refined
                best_score = refined_score
    except Exception:
        pass
    
    return best_pts, best_score


def construct():
    start_time = time.time()
    
    # Strategy: Use differential evolution to find a good configuration,
    # then refine with Nelder-Mead.
    
    # We work in a normalized space [0,1]^2 for each coordinate
    # The score is scale-invariant (ratio of areas), so normalization doesn't matter
    # but we need to ensure points are not all collinear or coincident.
    
    # Define the optimization bounds
    bounds = [(0, 1)] * (N * 2)
    
    best_pts = None
    best_score = 0.0
    
    # Run differential evolution with limited iterations
    def de_objective(x):
        p = x.reshape(N, 2)
        return -_score_points(p)
    
    elapsed = time.time() - start_time
    remaining = DEADLINE - elapsed
    
    if remaining > 5:
        try:
            # Use differential evolution with moderate settings
            de_result = differential_evolution(
                de_objective,
                bounds,
                seed=42,
                maxiter=100,
                popsize=15,
                tol=1e-6,
                mutation=(0.3, 1.0),
                recombination=0.7,
                init='sobol',
                polish=False,
                callback=None,
                x0=None
            )
            
            if de_result.fun < 0:
                candidate_pts = de_result.x.reshape(N, 2)
                candidate_score = _score_points(candidate_pts)
                if candidate_score > best_score:
                    best_pts = candidate_pts
                    best_score = candidate_score
        except Exception:
            pass
    
    # Try multiple random restarts with Nelder-Mead refinement
    elapsed = time.time() - start_time
    remaining = DEADLINE - elapsed
    
    # If we have time, try some structured initial configurations
    if best_pts is None or remaining > 10:
        # Try a regular grid-like configuration as starting point
        # 13 points: try to arrange them in a pattern that maximizes min triangle area
        # One approach: place points on a circle with some perturbation
        
        # Try circular arrangement
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        circle_pts = np.column_stack([0.5 + 0.45 * np.cos(angles), 
                                       0.5 + 0.45 * np.sin(angles)])
        circle_score = _score_points(circle_pts)
        
        if circle_score > best_score:
            best_pts = circle_pts
            best_score = circle_score
    
    # Now refine the best candidate with Nelder-Mead
    if best_pts is not None:
        elapsed = time.time() - start_time
        remaining = DEADLINE - elapsed
        if remaining > 5:
            best_pts, best_score = _refine_with_nelder_mead(best_pts, remaining)
    
    # Try a few more random restarts if time permits
    elapsed = time.time() - start_time
    remaining = DEADLINE - elapsed
    
    np.random.seed(123)
    restart_count = 0
    while remaining > 15:
        # Random restart
        random_pts = np.random.random((N, 2))
        random_score = _score_points(random_pts)
        
        if random_score > best_score:
            best_pts = random_pts
            best_score = random_score
        
        # Try to refine this random point
        if remaining > 20:
            refined_pts, refined_score = _refine_with_nelder_mead(random_pts, remaining - 5)
            if refined_score > best_score:
                best_pts = refined_pts
                best_score = refined_score
        
        restart_count += 1
        elapsed = time.time() - start_time
        remaining = DEADLINE - elapsed
    
    # If we still have no good result, return a default
    if best_pts is None:
        # Fallback: regular 13-gon
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_pts = np.column_stack([0.5 + 0.45 * np.cos(angles), 
                                     0.5 + 0.45 * np.sin(angles)])
    
    return best_pts
