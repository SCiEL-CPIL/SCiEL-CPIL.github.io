"""Heilbronn convex regions: maximize minimum triangle area over 13 points in unit-area convex region."""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time
import itertools

N = 13
DEADLINE = 140.0  # safety margin before 150s kill


def _triangle_areas(points):
    """Compute all C(n,3) triangle areas."""
    n = len(points)
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                cross = (points[j][0] - points[i][0]) * (points[k][1] - points[i][1]) - \
                        (points[j][1] - points[i][1]) * (points[k][0] - points[i][0])
                areas.append(0.5 * abs(cross))
    return np.array(areas)


def _convex_hull_area(points):
    """Compute convex hull area. Returns 0 if hull is degenerate."""
    try:
        hull = ConvexHull(points)
        return hull.volume  # in 2D, volume = area
    except Exception:
        return 0.0


def _score(points):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area <= 1e-12:
        return 0.0
    areas = _triangle_areas(points)
    return float(np.min(areas)) / hull_area


def _soft_min_objective(theta, temperature):
    """
    Objective: minimize -log-sum-exp of triangle areas / hull area,
    which approximates -min(areas)/hull_area as temperature -> 0.
    
    theta: 13 angles on unit circle (structural convexity)
    """
    n = len(theta)
    # Points on unit circle
    pts = np.column_stack([np.cos(theta), np.sin(theta)])
    
    hull_area = _convex_hull_area(pts)
    if hull_area <= 1e-12:
        return 1e10
    
    areas = _triangle_areas(pts)
    
    # Soft minimum: -log(sum(exp(-area/T))) / hull_area
    # We want to MAXIMIZE min(areas)/hull_area, so minimize -min(areas)/hull_area
    # Using log-sum-exp trick: min(a_i) ≈ -T * log(sum(exp(-a_i/T)))
    scaled = -areas / temperature
    # Shift for numerical stability
    max_scaled = np.max(scaled)
    log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min = -temperature * log_sum_exp
    
    return -soft_min / hull_area  # negative because we want to maximize


def _gradient_descent(theta0, temperature, max_iter=200, tol=1e-8):
    """L-BFGS-B optimization on angles."""
    try:
        result = minimize(
            _soft_min_objective,
            theta0,
            args=(temperature,),
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': tol, 'gtol': tol}
        )
        return result.x, result.fun
    except Exception:
        return theta0, _soft_min_objective(theta0, temperature)


def construct():
    start_time = time.time()
    
    # Initial best: regular 13-gon
    best_theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
    best_pts = np.column_stack([np.cos(best_theta), np.sin(best_theta)])
    best_score = _score(best_pts)
    
    # Multi-scale temperature annealing with multiple restarts
    temperatures = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    
    # Multiple random restarts
    n_restarts = 8
    rng = np.random.RandomState(42)
    
    for restart in range(n_restarts):
        if time.time() > DEADLINE:
            break
        
        # Perturb the current best or start fresh
        if restart == 0:
            theta0 = best_theta.copy()
        else:
            # Mix: sometimes perturb best, sometimes random
            if restart % 2 == 0:
                theta0 = best_theta + rng.normal(0, 0.1, N)
            else:
                theta0 = np.sort(rng.uniform(0, 2 * np.pi, N))
        
        # Sort angles to maintain ordering
        theta0 = np.sort(theta0)
        
        for temp in temperatures:
            if time.time() > DEADLINE:
                break
            
            # Scale iterations based on remaining time
            remaining = DEADLINE - time.time()
            max_iter = max(20, min(300, int(remaining * 2)))
            
            theta_opt, obj_val = _gradient_descent(theta0, temp, max_iter=max_iter)
            
            # Evaluate actual score
            pts = np.column_stack([np.cos(theta_opt), np.sin(theta_opt)])
            score = _score(pts)
            
            if score > best_score:
                best_score = score
                best_theta = theta_opt.copy()
                best_pts = pts.copy()
            
            # Use optimized theta as starting point for next (cooler) temperature
            theta0 = theta_opt
    
    # Final local refinement with very low temperature
    if time.time() < DEADLINE - 5:
        theta_opt, _ = _gradient_descent(best_theta, 0.001, max_iter=500)
        pts = np.column_stack([np.cos(theta_opt), np.sin(theta_opt)])
        score = _score(pts)
        if score > best_score:
            best_score = score
            best_pts = pts
    
    # Normalize to unit area: scale points so convex hull area = 1
    hull_area = _convex_hull_area(best_pts)
    if hull_area > 1e-12:
        scale = 1.0 / np.sqrt(hull_area)
        best_pts = best_pts * scale
    
    return best_pts
