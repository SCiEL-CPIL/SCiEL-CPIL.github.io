"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull
import itertools
import time

N = 14

# EVOLVE-BLOCK-START

def _min_triangle_area(pts):
    """Compute the minimum area over all triples of points."""
    n = len(pts)
    min_area = np.inf
    # Vectorized computation of all triangle areas
    # For each triple (i, j, k), area = 0.5 * |cross(pts[j]-pts[i], pts[k]-pts[i])|
    # We'll compute this in a batched way to avoid O(n^3) Python loops
    # But for n=14, n^3 = 2744, which is fine even in Python
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = pts[j] - pts[i]
                b = pts[k] - pts[i]
                area = 0.5 * abs(a[0] * b[1] - a[1] * b[0])
                if area < min_area:
                    min_area = area
    return min_area

def _convex_hull_area(pts):
    """Compute the area of the convex hull of the points."""
    try:
        hull = ConvexHull(pts)
        return hull.volume  # In 2D, volume is area
    except Exception:
        return 0.0

def _score(pts):
    """Compute the score: min triangle area / convex hull area."""
    hull_area = _convex_hull_area(pts)
    if hull_area <= 1e-12:
        return 0.0
    min_tri = _min_triangle_area(pts)
    return min_tri / hull_area

def _regular_polygon(n, radius=1.0):
    """Generate vertices of a regular n-gon inscribed in a circle of given radius."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return radius * np.column_stack((np.cos(angles), np.sin(angles)))

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # Start with a regular 14-gon as initial guess
    best_pts = _regular_polygon(N)
    best_score = _score(best_pts)
    
    # Define bounds for differential evolution
    # We'll optimize all 14 points in a square [-2, 2]^2
    # This gives enough room for the convex hull to form
    bounds = [(-2.0, 2.0)] * (2 * N)
    
    # Objective function for differential evolution
    def objective(x):
        pts = x.reshape(N, 2)
        return -_score(pts)  # Negate because DE minimizes
    
    # Run differential evolution
    try:
        de_result = differential_evolution(
            objective,
            bounds,
            seed=42,
            maxiter=200,
            popsize=15,
            tol=1e-8,
            mutation=(0.3, 1.0),
            recombination=0.8,
            polish=False,
            updating='deferred',
            callback=lambda xk, convergence: (time.time() > deadline)
        )
        
        if not (time.time() > deadline):
            candidate_pts = de_result.x.reshape(N, 2)
            candidate_score = _score(candidate_pts)
            if candidate_score > best_score:
                best_pts = candidate_pts
                best_score = candidate_score
    except Exception:
        pass
    
    # Local refinement using L-BFGS-B
    if time.time() < deadline:
        try:
            def neg_score(x):
                pts = x.reshape(N, 2)
                return -_score(pts)
            
            # Use multiple restarts from perturbed best solution
            rng = np.random.RandomState(42)
            for restart in range(5):
                if time.time() > deadline - 10.0:
                    break
                
                if restart == 0:
                    x0 = best_pts.flatten()
                else:
                    x0 = best_pts.flatten() + rng.normal(0, 0.1, size=2*N)
                
                result = minimize(
                    neg_score,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-2.0, 2.0)] * (2 * N),
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12}
                )
                
                if result.success or result.fun < 0:
                    candidate_pts = result.x.reshape(N, 2)
                    candidate_score = _score(candidate_pts)
                    if candidate_score > best_score:
                        best_pts = candidate_pts
                        best_score = candidate_score
        except Exception:
            pass
    
    # Final check: try a few more perturbations if time remains
    if time.time() < deadline - 5.0:
        rng = np.random.RandomState(123)
        for _ in range(20):
            if time.time() > deadline - 5.0:
                break
            perturbed = best_pts + rng.normal(0, 0.05, size=(N, 2))
            s = _score(perturbed)
            if s > best_score:
                best_pts = perturbed
                best_score = s
    
    return best_pts

# EVOLVE-BLOCK-END
