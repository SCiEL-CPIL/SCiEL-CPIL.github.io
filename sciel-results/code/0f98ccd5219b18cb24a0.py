"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time
import itertools
import random

N = 13

# EVOLVE-BLOCK-START

def _hull_area(points):
    """Compute the area of the convex hull of the given points."""
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is area
    except Exception:
        # Fallback: if hull computation fails, return a small value
        return 1e-10

def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by 3 points."""
    n = len(points)
    min_area = float('inf')
    for i, j, k in itertools.combinations(range(n), 3):
        p1, p2, p3 = points[i], points[j], points[k]
        # Area of triangle = 0.5 * |cross product|
        area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
        if area < min_area:
            min_area = area
    return min_area

def _score(points):
    """Score = min triangle area / convex hull area."""
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_tri = _min_triangle_area(points)
    return min_tri / hull_area

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 150s limit, 5s safety margin
    
    # Precompute triangle indices for speed
    tri_indices = list(itertools.combinations(range(N), 3))
    
    def _min_tri_fast(points):
        """Fast min triangle area using precomputed indices."""
        min_area = float('inf')
        for i, j, k in tri_indices:
            p1, p2, p3 = points[i], points[j], points[k]
            area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
            if area < min_area:
                min_area = area
        return min_area
    
    def _objective(points_flat):
        """Objective function for optimizer: negative score (to maximize)."""
        points = points_flat.reshape(N, 2)
        hull_area = _hull_area(points)
        if hull_area < 1e-10:
            return 1e10
        min_tri = _min_tri_fast(points)
        score = min_tri / hull_area
        return -score
    
    # Initial guess: regular polygon-like distribution
    def _initial_guess(seed):
        rng = random.Random(seed)
        # Try different initial configurations
        strategy = rng.randint(0, 2)
        if strategy == 0:
            # Fermat spiral (Vogel's model)
            i = np.arange(N)
            r = np.sqrt(i / N)
            theta = i * 2.399963229728653 + rng.random() * 2 * np.pi
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            points = np.column_stack((x, y))
        elif strategy == 1:
            # Regular polygon with slight perturbation
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False) + rng.random() * 0.1
            r = 1.0 + rng.random() * 0.2
            x = r * np.cos(angles)
            y = r * np.sin(angles)
            points = np.column_stack((x, y))
        else:
            # Random points in unit disk
            angles = rng.random() * 2 * np.pi
            r = np.sqrt(rng.random())
            x = r * np.cos(angles)
            y = r * np.sin(angles)
            points = np.column_stack((x, y))
        
        # Center the points
        points -= np.mean(points, axis=0)
        return points
    
    best_points = None
    best_score = 0.0
    
    # Try multiple initial configurations with local optimization
    for attempt in range(20):
        if time.time() > deadline - 5:
            break
        
        initial_points = _initial_guess(attempt * 7 + 1)
        
        # Use Powell method (derivative-free, robust for non-smooth objectives)
        try:
            result = minimize(
                _objective,
                initial_points.flatten(),
                method='Powell',
                options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            
            if result.success or result.fun < 0:
                opt_points = result.x.reshape(N, 2)
                s = _score(opt_points)
                if s > best_score:
                    best_score = s
                    best_points = opt_points.copy()
        except Exception:
            continue
        
        # Also try L-BFGS-B for gradient-based refinement
        if time.time() > deadline - 10:
            break
        try:
            result2 = minimize(
                _objective,
                initial_points.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10}
            )
            
            if result2.fun < 0:
                opt_points2 = result2.x.reshape(N, 2)
                s2 = _score(opt_points2)
                if s2 > best_score:
                    best_score = s2
                    best_points = opt_points2.copy()
        except Exception:
            continue
    
    if best_points is None:
        # Fallback to a reasonable initial guess
        i = np.arange(N)
        r = np.sqrt(i / N)
        theta = i * 2.399963229728653
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        best_points = np.column_stack((x, y))
        best_points -= np.mean(best_points, axis=0)
    
    return best_points

# EVOLVE-BLOCK-END
