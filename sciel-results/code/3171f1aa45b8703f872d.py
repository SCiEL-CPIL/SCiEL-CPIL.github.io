import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull
from itertools import combinations
import time

N = 13

def _min_triangle_area(points):
    """Compute the minimum triangle area over all triples of points."""
    n = len(points)
    min_area = np.inf
    # Vectorized calculation might be faster, but for N=13, loops are fine
    # There are 286 combinations
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Triangle area = 0.5 * |cross(pj-pi, pk-pi)|
                v1 = points[j] - points[i]
                v2 = points[k] - points[i]
                cross = v1[0] * v2[1] - v1[1] * v2[0]
                area = 0.5 * abs(cross)
                if area < min_area:
                    min_area = area
    return min_area

def _convex_hull_area(points):
    """Compute the area of the convex hull of the points."""
    try:
        hull = ConvexHull(points)
        return hull.area
    except Exception:
        # Degenerate case
        return 1e-10

def _score(points):
    """
    Score is min triangle area divided by convex hull area.
    """
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-9:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area

def _objective(x):
    """
    Objective function for minimization.
    x is a flat array of shape (2*N,).
    We want to maximize the score, so we minimize -score.
    """
    points = x.reshape(N, 2)
    s = _score(points)
    return -s

def construct():
    start_time = time.time()
    timeout = 120.0 # Safety margin
    
    # Initial guesses
    # 1. Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    init_reg = np.column_stack([np.cos(angles), np.sin(angles)]).flatten()
    
    # 2. Random points in unit square
    rng = np.random.RandomState(42)
    init_rand = rng.rand(N, 2).flatten()
    
    # 3. Mix of hull and interior
    # 7 points on circle, 6 inside
    angles_outer = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    outer = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
    angles_inner = np.linspace(0, 2 * np.pi, 6, endpoint=False) + np.pi/6
    inner = 0.5 * np.column_stack([np.cos(angles_inner), np.sin(angles_inner)])
    init_mix = np.vstack([outer, inner]).flatten()
    
    best_points = init_reg.reshape(N, 2)
    best_score = _score(best_points)
    
    # Differential Evolution
    # Bounds: [-5, 5] for each coordinate
    bounds = [(-5, 5)] * (2 * N)
    
    try:
        # DE is stochastic, set seed for reproducibility
        de_result = differential_evolution(
            _objective, 
            bounds, 
            seed=42, 
            maxiter=100, 
            popsize=15,
            tol=1e-8,
            init='latinhypercube'
        )
        
        if de_result.success or de_result.fun < -best_score:
            candidate_points = de_result.x.reshape(N, 2)
            candidate_score = _score(candidate_points)
            if candidate_score > best_score:
                best_score = candidate_score
                best_points = candidate_points
        
        elapsed = time.time() - start_time
        if elapsed < timeout - 10:
            # Polish with Powell
            # Powell is derivative-free and good for this
            res = minimize(
                _objective, 
                best_points.flatten(), 
                method='Powell',
                options={'maxiter': 500, 'disp': False}
            )
            if res.fun < -best_score:
                candidate_points = res.x.reshape(N, 2)
                candidate_score = _score(candidate_points)
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_points = candidate_points
                    
    except Exception:
        pass

    # Try a few random restarts with Powell if time permits
    elapsed = time.time() - start_time
    if elapsed < timeout - 20:
        for i in range(5):
            if time.time() - start_time > timeout - 5:
                break
            # Generate a new random start near the current best or completely random
            if i == 0:
                # Perturb best
                start = best_points + rng.randn(N, 2) * 0.1
            else:
                # Random in unit square
                start = rng.rand(N, 2)
            
            try:
                res = minimize(
                    _objective, 
                    start.flatten(), 
                    method='Powell',
                    options={'maxiter': 200, 'disp': False}
                )
                if res.fun < -best_score:
                    candidate_points = res.x.reshape(N, 2)
                    candidate_score = _score(candidate_points)
                    if candidate_score > best_score:
                        best_score = candidate_score
                        best_points = candidate_points
            except Exception:
                continue

    return best_points
