import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull

N = 14
DEADLINE = 110.0  # seconds, safety margin from 120s limit

def _triangle_areas(points):
    """Compute all C(n,3) triangle areas."""
    n = len(points)
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                ax, ay = points[i]
                bx, by = points[j]
                cx, cy = points[k]
                area = 0.5 * abs((bx - ax) * (cy - ay) - (by - ay) * (cx - ax))
                areas.append(area)
    return np.array(areas)

def _hull_area(points):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 1e-10

def _score(points):
    """Score = min triangle area / hull area."""
    areas = _triangle_areas(points)
    if len(areas) == 0:
        return 0.0
    min_area = np.min(areas)
    hull_a = _hull_area(points)
    if hull_a < 1e-15:
        return 0.0
    return min_area / hull_a

def _soft_min(areas, beta=50.0):
    """Soft-min approximation using log-sum-exp trick."""
    # soft_min(x) ≈ -log(sum(exp(-beta*x)))/beta for large beta
    # More stable: shift by min
    m = np.min(areas)
    shifted = areas - m
    # soft_min ≈ m - log(sum(exp(-beta*shifted)))/beta
    # For large beta, this approaches m
    exp_vals = np.exp(-beta * shifted)
    log_sum = np.log(np.sum(exp_vals))
    return m - log_sum / beta

def _objective_soft(x, beta=50.0):
    """Objective to minimize: -soft_min(triangle_areas) / hull_area"""
    points = x.reshape(-1, 2)
    areas = _triangle_areas(points)
    hull_a = _hull_area(points)
    if hull_a < 1e-15:
        return 1e10
    soft_m = _soft_min(areas, beta)
    return -soft_m / hull_a

def _regular_polygon(n):
    """Generate regular n-gon centered at origin with radius 1."""
    angles = 2 * np.pi * np.arange(n) / n
    return np.column_stack((np.cos(angles), np.sin(angles)))

def construct():
    start_time = time.time()
    
    # Start with regular 14-gon
    best_points = _regular_polygon(N)
    best_score = _score(best_points)
    
    # Try local refinement with Nelder-Mead
    def neg_score(x):
        points = x.reshape(-1, 2)
        return -_score(points)
    
    x0 = best_points.flatten()
    
    # Multiple restarts with different beta values for soft-min
    for beta in [30.0, 50.0, 100.0]:
        if time.time() - start_time > DEADLINE - 10:
            break
        
        def obj_beta(x):
            points = x.reshape(-1, 2)
            areas = _triangle_areas(points)
            hull_a = _hull_area(points)
            if hull_a < 1e-15:
                return 1e10
            soft_m = _soft_min(areas, beta)
            return -soft_m / hull_a
        
        try:
            res = minimize(obj_beta, x0, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
            if time.time() - start_time > DEADLINE - 5:
                break
            candidate = res.x.reshape(-1, 2)
            cand_score = _score(candidate)
            if cand_score > best_score:
                best_score = cand_score
                best_points = candidate
        except Exception:
            pass
    
    # Try a few random perturbations around the best point
    if time.time() - start_time < DEADLINE - 5:
        rng = np.random.RandomState(42)
        for _ in range(20):
            if time.time() - start_time > DEADLINE - 5:
                break
            noise = rng.normal(0, 0.01, (N, 2))
            candidate = best_points + noise
            cand_score = _score(candidate)
            if cand_score > best_score:
                best_score = cand_score
                best_points = candidate
                # Further refine
                try:
                    res = minimize(neg_score, candidate.flatten(), method='Nelder-Mead',
                                  options={'maxiter': 200, 'xatol': 1e-9, 'fatol': 1e-11})
                    if time.time() - start_time > DEADLINE - 3:
                        break
                    candidate2 = res.x.reshape(-1, 2)
                    cand_score2 = _score(candidate2)
                    if cand_score2 > best_score:
                        best_score = cand_score2
                        best_points = candidate2
                except Exception:
                    pass
    
    # Final check
    if time.time() - start_time > DEADLINE - 2:
        pass  # Just return what we have
    
    return best_points
