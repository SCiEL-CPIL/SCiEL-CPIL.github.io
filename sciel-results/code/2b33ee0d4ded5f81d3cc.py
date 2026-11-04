"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from itertools import combinations


def construct():
    """
    Place 13 points in a planar convex region normalized to unit area to maximize
    the minimum area over all triples.
    
    Approach:
    1. Parameterize points by angles (sorted) and radii to enforce convex position.
    2. Use logarithmic barrier for convexity: enforce positive cross products.
    3. Optimize to maximize minimum triangle area / hull area.
    """
    
    # Generate regular 13-gon centered at origin as initial guess
    angles_init = np.linspace(0, 2 * np.pi, N, endpoint=False)
    points_init = np.column_stack([np.cos(angles_init), np.sin(angles_init)])
    
    def compute_hull_area(pts):
        """Compute convex hull area using shoelace formula on sorted points."""
        # Sort by angle around centroid
        centroid = pts.mean(axis=0)
        angles = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
        sorted_idx = np.argsort(angles)
        pts_sorted = pts[sorted_idx]
        
        # Shoelace formula
        n = len(pts_sorted)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += pts_sorted[i, 0] * pts_sorted[j, 1]
            area -= pts_sorted[j, 0] * pts_sorted[i, 1]
        return 0.5 * abs(area)
    
    def min_triangle_area(pts):
        """Compute minimum triangle area among all triples."""
        n = len(pts)
        min_area = float('inf')
        for i, j, k in combinations(range(n), 3):
            a = pts[i]
            b = pts[j]
            c = pts[k]
            area = 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]))
            if area < min_area:
                min_area = area
        return min_area
    
    def is_convex_position(pts):
        """Check if points are in convex position using cross products."""
        # Sort by angle around centroid
        centroid = pts.mean(axis=0)
        angles = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
        sorted_idx = np.argsort(angles)
        pts_sorted = pts[sorted_idx]
        
        n = len(pts_sorted)
        for i in range(n):
            p1 = pts_sorted[i]
            p2 = pts_sorted[(i + 1) % n]
            p3 = pts_sorted[(i + 2) % n]
            cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1])
            if cross <= 1e-10:
                return False
        return True
    
    def get_sorted_points(pts):
        """Sort points by angle around centroid."""
        centroid = pts.mean(axis=0)
        angles = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
        sorted_idx = np.argsort(angles)
        return pts[sorted_idx]
    
    def convexity_barrier(pts, mu=0.01):
        """
        Compute logarithmic barrier for convexity.
        Returns (barrier_value, gradient_info) or inf if not convex.
        """
        pts_sorted = get_sorted_points(pts)
        n = len(pts_sorted)
        
        barrier_val = 0.0
        for i in range(n):
            p1 = pts_sorted[i]
            p2 = pts_sorted[(i + 1) % n]
            p3 = pts_sorted[(i + 2) % n]
            cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1])
            if cross <= 1e-12:
                return float('inf'), None
            barrier_val -= mu * np.log(cross)
        
        return barrier_val, pts_sorted
    
    # Objective function with barrier for convexity
    def objective(x):
        pts = x.reshape(N, 2)
        
        # Get barrier value
        barrier_val, pts_sorted = convexity_barrier(pts, mu=0.05)
        if np.isinf(barrier_val):
            return 1e10
        
        hull_area = compute_hull_area(pts)
        if hull_area < 1e-10:
            return 1e10
        
        mt_area = min_triangle_area(pts)
        
        # Maximize mt_area / hull_area, add barrier penalty
        return -(mt_area / hull_area) + barrier_val
    
    # Start from regular polygon
    x0 = points_init.flatten()
    
    # Use SLSQP optimization with multiple restarts
    best_result = None
    best_fun = float('inf')
    
    # Try multiple initializations
    for trial in range(3):
        if trial == 0:
            x_trial = points_init.flatten()
        elif trial == 1:
            # Perturbed regular polygon
            noise = np.random.randn(N, 2) * 0.1
            x_trial = (points_init + noise).flatten()
        else:
            # Slightly different angle spacing
            angles_trial = np.linspace(0, 2 * np.pi, N, endpoint=False) + np.random.randn(N) * 0.05
            radii = 1.0 + np.random.randn(N) * 0.05
            pts_trial = np.column_stack([np.cos(angles_trial) * radii, np.sin(angles_trial) * radii])
            x_trial = pts_trial.flatten()
        
        result = minimize(
            objective,
            x_trial,
            method='SLSQP',
            options={'maxiter': 500, 'ftol': 1e-10, 'disp': False}
        )
        
        if result.fun < best_fun:
            best_fun = result.fun
            best_result = result
    
    if best_result is None or best_result.fun > 1e9:
        optimized_points = points_init
    else:
        optimized_points = best_result.x.reshape(N, 2)
    
    # If optimization failed, fall back to regular polygon
    if optimized_points.shape != (N, 2):
        optimized_points = points_init
    
    # Normalize: center at origin and scale so convex hull area = 1
    hull_area = compute_hull_area(optimized_points)
    if hull_area > 1e-10:
        scale = 1.0 / np.sqrt(hull_area)
        centroid = optimized_points.mean(axis=0)
        optimized_points = (optimized_points - centroid) * scale
    
    return optimized_points
# EVOLVE-BLOCK-END
