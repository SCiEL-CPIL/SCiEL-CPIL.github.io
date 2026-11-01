"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull
import itertools

N = 14

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115  # 115s safety margin before 120s kill
    
    def get_points(x):
        """Reconstruct points from parameter vector.
        Fix p0 at origin, p1 at (1,0) to remove translation/rotation invariance.
        x has 2*(N-2) = 24 elements for points p2..p13."""
        pts = np.zeros((N, 2))
        pts[0] = [0.0, 0.0]
        pts[1] = [1.0, 0.0]
        for i in range(N - 2):
            pts[i + 2, 0] = x[2 * i]
            pts[i + 2, 1] = x[2 * i + 1]
        return pts
    
    def triangle_areas(pts):
        """Compute all triangle areas from N points."""
        n = len(pts)
        areas = np.array([
            0.5 * abs(np.cross(pts[j] - pts[i], pts[k] - pts[i]))
            for i in range(n) for j in range(i+1, n) for k in range(j+1, n)
        ])
        return areas
    
    def hull_area(pts):
        """Compute convex hull area."""
        try:
            hull = ConvexHull(pts)
            return hull.volume  # In 2D, volume is area
        except Exception:
            return 1e-10
    
    def objective(x):
        """Negative of (min_triangle_area / hull_area) for minimization."""
        pts = get_points(x)
        areas = triangle_areas(pts)
        min_area = np.min(areas)
        h_area = hull_area(pts)
        if h_area < 1e-10:
            return 1e10
        return -(min_area / h_area)
    
    def soft_min_objective(x, beta=50.0):
        """Smooth surrogate using log-sum-exp for soft-min."""
        pts = get_points(x)
        areas = triangle_areas(pts)
        h_area = hull_area(pts)
        if h_area < 1e-10:
            return 1e10
        # Soft min: -log(sum(exp(-beta*a))) / beta approximates min(a)
        # We want to maximize min(a)/h_area, so minimize -soft_min(a)/h_area
        log_sum = np.log(np.sum(np.exp(-beta * areas)))
        soft_min_val = -log_sum / beta
        return -(soft_min_val / h_area)
    
    # Initial guess: regular 14-gon centered at origin, then shift/rotate to fix p0, p1
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    pts_init = np.column_stack((np.cos(angles), np.sin(angles)))
    
    # Center at origin
    pts_init -= np.mean(pts_init, axis=0)
    
    # Find p0 and p1 (first two points), compute translation and rotation to fix them
    # We want p0 -> (0,0) and p1 -> (1,0)
    p0 = pts_init[0].copy()
    p1 = pts_init[1].copy()
    
    # Translate so p0 is at origin
    pts_init = pts_init - p0
    
    # Now p0 is at origin. Rotate so p1 is along positive x-axis.
    p1_trans = pts_init[1].copy()
    angle = np.arctan2(p1_trans[1], p1_trans[0])
    R = np.array([[np.cos(-angle), -np.sin(-angle)],
                  [np.sin(-angle),  np.cos(-angle)]])
    pts_init = (R @ pts_init.T).T
    
    # Now p0 should be ~(0,0) and p1 should be ~(|p1|, 0). Scale so p1 = (1,0)
    scale = 1.0 / np.linalg.norm(pts_init[1])
    pts_init *= scale
    
    # Extract free parameters (p2..p13)
    x0 = np.array([pts_init[i, j] for i in range(2, N) for j in range(2)])
    
    # Bounds for differential evolution: points should be in a reasonable range
    bounds = [(-2.0, 2.0)] * (2 * (N - 2))
    
    best_x = x0.copy()
    best_score = -objective(x0)
    
    # Try differential evolution first (global search)
    if time.time() < deadline - 60:
        try:
            result = differential_evolution(
                objective, bounds,
                x0=x0,
                maxiter=100,
                popsize=15,
                tol=1e-6,
                mutation=(0.3, 1.0),
                recombination=0.7,
                seed=42,
                polish=True,
                updating='deferred'
            )
            if result.fun < -best_score:
                best_score = -result.fun
                best_x = result.x
        except Exception:
            pass
    
    # Local refinement with L-BFGS-B
    if time.time() < deadline - 20:
        try:
            res = minimize(
                objective, best_x,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res.fun < -best_score:
                best_score = -res.fun
                best_x = res.x
        except Exception:
            pass
    
    # Try soft-min objective for smoother landscape
    if time.time() < deadline - 10:
        try:
            res = minimize(
                lambda x: soft_min_objective(x, beta=50.0), best_x,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12}
            )
            # Evaluate with exact objective
            exact_score = -objective(res.x)
            if exact_score > best_score:
                best_score = exact_score
                best_x = res.x
        except Exception:
            pass
    
    # Final polish with tighter tolerance
    if time.time() < deadline - 5:
        try:
            res = minimize(
                objective, best_x,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            if res.fun < -best_score:
                best_score = -res.fun
                best_x = res.x
        except Exception:
            pass
    
    # Reconstruct final points
    pts = get_points(best_x)
    
    # Normalize: the score is min_triangle_area / hull_area
    # But the evaluator normalizes by convex hull area, so we just return the points
    # The evaluator will compute the score itself
    
    return pts
# EVOLVE-BLOCK-END

