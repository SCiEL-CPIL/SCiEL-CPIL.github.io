"""Optimized program for Heilbronn convex regions n=14."""
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 14

def _triangle_areas_from_angles(angles):
    """Compute all triangle areas for points on unit circle at given angles."""
    n = len(angles)
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Area of triangle with vertices on unit circle at angles ai, aj, ak
                # Using cross product formula
                ai, aj, ak = angles[i], angles[j], angles[k]
                area = 0.5 * abs(np.sin(aj - ai) + np.sin(ak - aj) + np.sin(ai - ak))
                areas.append(area)
    return np.array(areas)

def _convex_hull_area_from_angles(angles):
    """Compute area of convex polygon with vertices on unit circle at given angles."""
    n = len(angles)
    # Sort angles to get polygon in order
    sorted_angles = np.sort(angles)
    # Area of polygon inscribed in unit circle
    area = 0.0
    for i in range(n):
        a1 = sorted_angles[i]
        a2 = sorted_angles[(i + 1) % n]
        area += 0.5 * np.sin(a2 - a1)
    return abs(area)

def _score_from_angles(angles):
    """Compute the score: min triangle area / convex hull area."""
    min_tri = np.min(_triangle_areas_from_angles(angles))
    hull_area = _convex_hull_area_from_angles(angles)
    if hull_area < 1e-12:
        return 0.0
    return min_tri / hull_area

def _objective(angles):
    """Negative score for minimization."""
    return -_score_from_angles(angles)

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin
    
    # Start with regular polygon (equally spaced angles)
    regular_angles = 2 * np.pi * np.arange(N) / N
    
    best_angles = regular_angles.copy()
    best_score = _score_from_angles(best_angles)
    
    # Try multiple starting points with differential evolution
    n_trials = 0
    max_trials = 10
    
    while n_trials < max_trials and time.time() < deadline - 15.0:
        n_trials += 1
        
        # Random perturbation of regular angles
        rng = np.random.default_rng(n_trials * 1000 + N)
        if n_trials == 1:
            init_angles = regular_angles + rng.normal(0, 0.1, N)
        else:
            init_angles = regular_angles + rng.normal(0, 0.3, N)
        
        # Normalize angles to [0, 2*pi)
        init_angles = init_angles % (2 * np.pi)
        
        # Use differential evolution for global optimization
        try:
            result = differential_evolution(
                _objective,
                bounds=[(0, 2 * np.pi)] * N,
                seed=n_trials * 100 + N,
                maxiter=50,
                popsize=10,
                tol=1e-6,
                polish=True
            )
            
            trial_score = -result.fun
            if trial_score > best_score:
                best_score = trial_score
                best_angles = result.x.copy()
                
        except Exception:
            pass
        
        # Check time
        if time.time() > deadline - 20.0:
            break
    
    # Convert angles to points on unit circle
    points = np.column_stack([np.cos(best_angles), np.sin(best_angles)])
    
    return points
